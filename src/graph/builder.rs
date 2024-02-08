use itertools::Itertools;

use super::*;
use std::{self, cmp::max, iter::Step, ops::Range};

#[derive(Debug, Default, Clone)]
pub struct GraphBuilder {
    pub a: NodeA,
    pub b: NodeB,
    pub connections_a: VecA<Vec<NodeB>>,
    pub connections_b: VecB<Vec<NodeA>>,
    /// The number of crossings from merging twins.
    pub self_crossings: u64,
}

impl Graph {
    pub fn builder(&self) -> GraphBuilder {
        GraphBuilder {
            a: self.a,
            b: self.b,
            connections_a: self.connections_a.clone(),
            connections_b: self.connections_b.clone(),
            self_crossings: self.self_crossings,
        }
    }
}

impl GraphBuilder {
    pub fn with_sizes(a: NodeA, b: NodeB) -> Self {
        Self {
            a,
            b,
            connections_a: VecA::new(a),
            connections_b: VecB::new(b),
            self_crossings: 0,
        }
    }
    /// Returns the id of the pushed node (not the length).
    pub fn push_node_a(&mut self) -> NodeA {
        let id = self.connections_a.push();
        self.a = self.connections_a.len();
        id
    }

    /// Returns the id of the pushed node (not the length).
    pub fn push_node_b(&mut self) -> NodeB {
        let id = self.connections_b.push();
        self.b = self.connections_b.len();
        id
    }

    pub fn push_edge(&mut self, a: NodeA, b: NodeB) {
        self[a].push(b);
        self[b].push(a);
    }

    pub fn try_push_edge(&mut self, a: NodeA, b: NodeB) -> bool {
        if self[a].contains(&b) {
            false
        } else {
            self.push_edge(a, b);
            true
        }
    }

    pub fn drop_a(&mut self, a: &[NodeA]) {
        let mut a = a.to_vec();
        a.sort();
        for &a in a.iter().rev() {
            self.connections_a.remove(a.0);
        }
        self.reconstruct_b();
    }

    pub fn drop_b(&mut self, b: &[NodeB]) {
        let mut b = b.to_vec();
        b.sort();
        for &b in b.iter().rev() {
            self.connections_b.remove(b.0);
        }
        self.reconstruct_a();
    }

    pub fn new(connections_b: VecB<Vec<NodeA>>) -> GraphBuilder {
        let a = Step::forward(
            *connections_b
                .iter()
                .filter_map(|x| x.iter().max())
                .max()
                .unwrap(),
            1,
        );
        let mut g = Self {
            a,
            b: connections_b.len(),
            connections_a: Default::default(),
            connections_b,
            self_crossings: 0,
        };
        g.reconstruct_a();
        g
    }

    pub fn to_quick_graph(&self) -> Graph {
        Graph {
            a: self.a,
            b: self.b,
            m: self.connections_a.iter().map(|x| x.len()).sum::<usize>(),
            b_permutation: Default::default(),
            connections_a: self.connections_a.clone(),
            connections_b: self.connections_b.clone(),
            crossings: None,
            reduced_crossings: None,
            intervals: self.intervals(),
            self_crossings: self.self_crossings,
            must_come_before: VecB::new(self.b),
        }
    }

    /// Split into non-trivial graphs corresponding to disjoint intervals.
    fn split(&self) -> Vec<GraphBuilder> {
        let mut intervals = self.intervals();
        // Suffix min of start
        {
            let mut min_start = self.a;
            for r in intervals.iter_mut().rev() {
                min_start = min(min_start, r.start);
                r.start = min_start;
            }
        }
        // Prefix max of end
        {
            let mut max_end = NodeA(0);
            for r in intervals.iter_mut() {
                max_end = max(max_end, r.end);
                r.end = max_end;
            }
        }

        let mut graphs = vec![];
        let mut i = 0;
        while i < self.b.0 {
            // Find the component starting at intervals[i].
            let mut j = i + 1;
            let mut end = intervals[NodeB(i)].end;
            while j < self.b.0 && intervals[NodeB(j)].start < end {
                end = max(end, intervals[NodeB(j)].end);
                j += 1;
            }
            // Size-1 components can be skipped.
            if j == i + 1 {
                i = j;
                continue;
            }
            // Convert the component into a Graph on its own.
            let g = GraphBuilder::new(VecB {
                v: self.connections_b.v[i..j].to_vec(),
            });
            graphs.push(g);
            i = j;
        }
        graphs
    }

    fn to_graph(&self) -> Graph {
        let (crossings, reduced_crossings) = self.crossings();
        Graph {
            a: self.a,
            b: self.b,
            m: self.connections_a.iter().map(|x| x.len()).sum::<usize>(),
            b_permutation: Default::default(),
            connections_a: self.connections_a.clone(),
            connections_b: self.connections_b.clone(),
            crossings: Some(crossings),
            reduced_crossings: Some(reduced_crossings),
            intervals: self.intervals(),
            self_crossings: self.self_crossings,
            must_come_before: self.find_siblings(),
        }
    }

    /// Permute the vertices of the graph so that B is roughly increasing.
    /// TODO: Store metadata to be able to invert the final solution.
    /// TODO: Allow customizing whether crossings are built.
    pub fn build(mut self) -> Graphs {
        self.drop_singletons();
        self.merge_twins();
        self.sort_edges();
        self.permute(initial_solution(&self.to_quick_graph()));
        self.sort_edges();
        self.split().iter().map(|g| g.to_graph()).collect()
    }
}

impl GraphBuilder {
    fn sort_edges(&mut self) {
        for l in self.connections_a.iter_mut() {
            l.sort();
        }
        for l in self.connections_b.iter_mut() {
            l.sort();
        }
    }

    /// Drop all nodes of degree 0.
    fn drop_singletons(&mut self) {
        self.connections_b.retain(|b| !b.is_empty());
        self.reconstruct_a();
        self.connections_a.retain(|a| !a.is_empty());
        self.reconstruct_b();
        self.sort_edges();
        let da = Step::steps_between(&self.connections_a.len(), &self.a).unwrap();
        let db = Step::steps_between(&self.connections_b.len(), &self.b).unwrap();
        // if da > 0 || db > 0 {
        //     eprintln!("Dropped {da} and {db} singletons",);
        // }
    }

    /// Merge vertices with the same set of neighbours.
    fn merge_twins(&mut self) {
        self.sort_edges();
        self.connections_b.sort();
        let mut self_crossings = 0;
        self.connections_b = VecB {
            v: (NodeB(0)..self.b)
                .group_by(|x| self[*x].clone())
                .into_iter()
                .map(|(key, group)| {
                    let cnt = group.into_iter().count();
                    if cnt == 1 {
                        return key;
                    }
                    let pairs = cnt * (cnt - 1) / 2;
                    let incr = Self::edge_list_crossings(&key, &key) * pairs as u64;
                    self_crossings += incr;
                    // eprintln!("Keys: {:?} cnt {cnt} incr {incr}", key);
                    key.iter()
                        .flat_map(|x| std::iter::repeat(*x).take(cnt))
                        .collect()
                })
                .collect(),
        };
        self.self_crossings += self_crossings;
        // eprintln!(
        //     "Merged {} twins; {self_crossings} self crossings",
        //     Step::steps_between(&self.connections_b.len(), &self.b).unwrap()
        // );
        self.reconstruct_a();
    }

    /// Count pairs (u,v) such that u must always be left of v:
    /// 0. Exclude pairs with disjoint intervals; they are not interesting.
    /// 1. cvw < cwv => cuw <= cwu (if v before w, than so u)
    /// 2. cwu < cuw => cwv <= cvw (if u after w, than so v)
    /// 3. cuv <= cvu
    fn find_siblings(&self) -> VecB<Vec<NodeB>> {
        // eprint!("Siblings..\r");
        let intervals = self.intervals();
        let c = self.crossings().0;
        let mut siblings = 0;
        let mut rev_siblings = 0;
        let mut rev_sibling_weight = 0;
        // For each node, the other nodes that must come before it.
        let mut must_come_before: VecB<Vec<NodeB>> = VecB::new(self.b);
        for u in NodeB(0)..self.b {
            'pairs: for v in NodeB(0)..self.b {
                if u == v {
                    continue;
                }
                // Disjoint intervals?
                if intervals[u].end <= intervals[v].start || intervals[v].end <= intervals[u].start
                {
                    continue;
                }
                for w in NodeB(0)..self.b {
                    if c[v][w] < c[w][v] && c[u][w] > c[w][u] {
                        continue 'pairs;
                    }
                    if c[w][u] < c[u][w] && c[w][v] > c[v][w] {
                        continue 'pairs;
                    }
                }
                // if c[u][v] < c[v][u] || (c[u][v] == c[v][u] && u < v) {
                if c[u][v] < c[v][u] {
                    siblings += 1;
                    must_come_before[v].push(u);
                } else {
                    rev_siblings += 1;
                    rev_sibling_weight += c[u][v] - c[v][u];
                }
            }
        }
        // eprintln!("Found {siblings} siblings");
        // eprintln!("Surprises: {rev_siblings} of total score {rev_sibling_weight}");
        must_come_before
    }

    /// Reconstruct `connections_a`, given `connections_b`.
    pub fn reconstruct_a(&mut self) {
        self.a = self.connections_b.a_len();
        self.b = self.connections_b.len();
        self.connections_a = VecA::new(self.a);
        for b in NodeB(0)..self.b {
            for &a in self.connections_b[b].iter() {
                self.connections_a[a].push(b);
            }
        }
    }

    /// Reconstruct `connections_b`, given `connections_a`.
    fn reconstruct_b(&mut self) {
        self.a = self.connections_a.len();
        self.b = self.connections_a.b_len();
        self.connections_b = VecB::new(self.b);
        for a in NodeA(0)..self.a {
            for &b in self.connections_a[a].iter() {
                self.connections_b[b].push(a);
            }
        }
    }

    /// Permute the nodes of B such that the given solution is simply 0..b.
    fn permute(&mut self, solution: Solution) {
        self.connections_b = VecB {
            v: solution
                .iter()
                .map(|&b| std::mem::take(&mut self.connections_b[b]))
                .collect(),
        };
        self.reconstruct_a();
    }

    fn intervals(&self) -> VecB<Range<NodeA>> {
        VecB {
            v: self
                .connections_b
                .iter()
                .map(|b| (*b.iter().min().unwrap()..*b.iter().max().unwrap()))
                .collect(),
        }
    }

    pub fn edge_list_crossings(e1: &Vec<NodeA>, e2: &Vec<NodeA>) -> u64 {
        let mut result: u64 = 0;
        for edge_i in e1 {
            for edge_j in e2 {
                if edge_i > edge_j {
                    result += 1;
                }
            }
        }
        result
    }

    pub fn one_node_crossings(&self, i: NodeB, j: NodeB) -> u64 {
        Self::edge_list_crossings(&self[i], &self[j])
    }

    fn crossings(&self) -> (VecB<VecB<u64>>, VecB<VecB<u64>>) {
        // eprintln!("Crossings..");
        let mut crossings: VecB<VecB<u64>> = VecB {
            v: vec![VecB::new(self.b); self.b.0],
        };
        for node_i in NodeB(0)..self.b {
            for node_j in NodeB(0)..self.b {
                crossings[node_i][node_j] = self.one_node_crossings(node_i, node_j);
            }
        }
        let mut reduced_crossings = VecB {
            v: vec![VecB::new(self.b); self.b.0],
        };
        for i in NodeB(0)..self.b {
            for j in NodeB(0)..self.b {
                reduced_crossings[i][j] = crossings[i][j].saturating_sub(crossings[j][i]);
            }
        }
        (crossings, reduced_crossings)
    }
}

impl Index<NodeA> for GraphBuilder {
    type Output = Vec<NodeB>;

    fn index(&self, index: NodeA) -> &Self::Output {
        &self.connections_a[index]
    }
}

impl IndexMut<NodeA> for GraphBuilder {
    fn index_mut(&mut self, index: NodeA) -> &mut Self::Output {
        &mut self.connections_a[index]
    }
}

impl Index<NodeB> for GraphBuilder {
    type Output = Vec<NodeA>;

    fn index(&self, index: NodeB) -> &Self::Output {
        &self.connections_b[index]
    }
}

impl IndexMut<NodeB> for GraphBuilder {
    fn index_mut(&mut self, index: NodeB) -> &mut Self::Output {
        &mut self.connections_b[index]
    }
}
