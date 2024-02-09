use itertools::Itertools;

use super::*;
use std::{
    self,
    cmp::max,
    iter::{zip, Step},
    ops::Range,
};

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
        let mut size_1 = 0;
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
                size_1 += 1;
                i = j;
                continue;
            }
            let start = intervals[NodeB(i)].start.0;
            // Convert the component into a Graph on its own.
            let mut new_cb = self.connections_b.v[i..j].to_vec();
            for l in &mut new_cb {
                for x in l {
                    *x = NodeA(x.0 - start);
                }
            }
            let g = GraphBuilder::new(VecB { v: new_cb });
            graphs.push(g);
            i = j;
        }
        info!("Singleton parts: {}", size_1);
        info!(
            "Split into parts with sizes: {:?}",
            graphs.iter().map(|g| (g.a, g.b)).collect::<Vec<_>>()
        );
        graphs
    }

    pub fn num_edges(&self) -> usize {
        self.connections_a.iter().map(|x| x.len()).sum::<usize>()
    }

    fn to_graph(mut self) -> Graph {
        let (crossings, reduced_crossings) = self.crossings();
        let mut must_come_before = self.find_siblings();
        let must_come_before_2 = self.dominating_pairs();
        for (a, b) in must_come_before.iter_mut().zip(must_come_before_2.iter()) {
            a.extend(b);
        }

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
            must_come_before,
        }
    }

    /// Print statistics on the graph.
    fn print_stats(&self) {
        info!("A={:?} B={:?}, {} edges", self.a, self.b, self.num_edges());

        if !log::log_enabled!(log::Level::Debug) {
            return;
        }

        let degs = self.connections_b.iter().map(|x| x.len()).join(",");
        debug!("Degrees:\n{degs}");

        let mut connections = self.connections_b.clone();
        connections.sort_by_key(|x| {
            assert!(x.is_sorted());
            (
                // First by length 0 and 1.
                (x.last().unwrap().0 - x.first().unwrap().0).min(2),
                // Then by start pos.
                x.first().unwrap().0,
                // Then by end pos.
                x.last().unwrap().0,
            )
        });

        let mut rows: Vec<Vec<Vec<NodeA>>> = vec![];
        for cs in connections.iter() {
            let first = cs.first().unwrap();
            let row = rows
                .iter_mut()
                .find(|r| r.last().is_some_and(|l| l.last().unwrap().0 + 1 < first.0));
            let row = match row {
                Some(row) => row,
                None => {
                    rows.push(vec![]);
                    rows.last_mut().unwrap()
                }
            };
            row.push(cs.clone());
        }
        debug!("Neighbours:");
        for row in rows {
            let mut line = vec![b' '; self.a.0];
            for cs in row {
                line[cs.first().unwrap().0..=cs.last().unwrap().0].fill(b'-');
                for c in cs {
                    line[c.0] = match line[c.0] {
                        b'-' => b'x',
                        b'x' => b'2',
                        c => c + 1,
                    };
                }
            }
            debug!("{}", String::from_utf8_lossy(&line));
        }
    }

    /// Permute the vertices of the graph so that B is roughly increasing.
    /// TODO: Store metadata to be able to invert the final solution.
    /// TODO: Allow customizing whether crossings are built.
    pub fn build(mut self) -> Graphs {
        self.drop_singletons();
        self.merge_twins();
        self.merge_adjacent_edges();
        self.sort_edges();
        self.permute(initial_solution(&self.to_quick_graph()));
        self.sort_edges();
        self.print_stats();
        self.split().into_iter().map(|g| g.to_graph()).collect()
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
        let a_old = self.a;
        let b_old = self.b;
        self.connections_b.retain(|b| !b.is_empty());
        self.reconstruct_a();
        self.connections_a.retain(|a| !a.is_empty());
        self.reconstruct_b();
        self.sort_edges();
        let da = Step::steps_between(&self.a, &a_old).unwrap();
        let db = Step::steps_between(&self.b, &b_old).unwrap();
        if da > 0 || db > 0 {
            info!("Dropped {da} and {db} singletons",);
        }
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
                    key.iter()
                        .flat_map(|x| std::iter::repeat(*x).take(cnt))
                        .collect()
                })
                .collect(),
        };
        self.self_crossings += self_crossings;
        info!(
            "Merged {} twins; {self_crossings} self crossings",
            Step::steps_between(&self.connections_b.len(), &self.b).unwrap()
        );
        self.reconstruct_a();
    }

    /// Given two adjacent nodes (u,v) in A.
    /// If the nbs of u and v are *only* connected to u and v, then we can merge u and v, and we can merge their neighbours.
    fn merge_adjacent_edges(&mut self) {
        self.sort_edges();

        let mut merged = 0;
        for ((x, cl), (y, cr)) in (NodeA(0)..self.a)
            .zip(self.connections_a.iter())
            .filter(|(_, cl)| !cl.is_empty())
            .tuple_windows()
        {
            if cl.is_empty() || cr.is_empty() {
                continue;
            }
            // Single nb each.
            if cl.first() != cl.last() {
                continue;
            }
            if cr.first() != cr.last() {
                continue;
            }

            let u = cl.first().unwrap();
            let v = cr.first().unwrap();

            if self[*u].is_empty() || self[*v].is_empty() {
                continue;
            }

            // Those nbs must only connect to cl and cr.
            if self[*u].first() != self[*u].last() {
                continue;
            }
            if self[*v].first() != self[*v].last() {
                continue;
            }

            assert_eq!(self[*u][0], x);
            assert_eq!(self[*v][0], y);

            // Merge u into v.
            assert!(self[y].len() == self[*v].len());
            let l = self[*u].len();
            self.connections_b[*u].clear();
            self.connections_b[*v].extend((0..l).map(|_| y));
            merged += 1;
        }

        info!("Merged {merged} adjacent edges",);
        self.reconstruct_a();
        self.drop_singletons();
    }

    /// Find pairs (u,v) with equal degree and neighbours(u) <= neighbours(v).
    fn dominating_pairs(&mut self) -> VecB<Vec<NodeB>> {
        // For each node, the other nodes that must come before it.
        let mut must_come_before: VecB<Vec<NodeB>> = VecB::new(self.b);
        if get_flag("no_dominating_pairs") {
            return must_come_before;
        }

        let mut dominating_pairs = 0;
        let mut strong_dominating_pairs = 0;
        let mut stronger_dominating_pairs = 0;

        self.sort_edges();
        for u in NodeB(0)..self.b {
            for v in NodeB(0)..self.b {
                if u == v {
                    continue;
                }
                // Already handled elsewhere.
                if self[u].last() < self[v].first() {
                    continue;
                }
                if self[u].len() != self[v].len() {
                    if get_flag("no_strong_dominating_pairs") {
                        continue;
                    }
                    if self[u].len() < self[v].len() {
                        // u < v if nbs(u) <= the first u.len() of nbs(v).
                        if zip(&self[u], &self[v]).all(|(x, y)| x <= y) {
                            must_come_before[v].push(u);
                            strong_dominating_pairs += 1;
                            continue;
                        }
                    } else {
                        // u < v if the last v.len() of nbs(u) <= nbs(v).
                        if zip(self[u].iter().rev(), self[v].iter().rev()).all(|(x, y)| x <= y) {
                            must_come_before[v].push(u);
                            strong_dominating_pairs += 1;
                            continue;
                        }
                    }
                    if get_flag("no_stronger_dominating_pairs") {
                        continue;
                    }
                    if (0..self[u].len()).all(|i| {
                        // u[i] must be smaller than v[j]
                        // for all j>=floor(i*vl/ul)
                        let j = (i * self[v].len()) / self[u].len();
                        self[u][i] < self[v][j]
                    }) {
                        must_come_before[v].push(u);
                        stronger_dominating_pairs += 1;
                    }
                    continue;
                }
                if self[u] == self[v] {
                    if u < v {
                        must_come_before[v].push(u);
                    }
                } else if zip(&self[u], &self[v]).all(|(x, y)| x <= y) {
                    must_come_before[v].push(u);
                    dominating_pairs += 1;
                }
            }
        }
        info!("Found {dominating_pairs} dominating pairs");
        info!("Found {strong_dominating_pairs} strong dominating pairs");
        info!("Found {stronger_dominating_pairs} stronger dominating pairs");
        must_come_before
    }

    /// Count pairs (u,v) such that u must always be left of v:
    /// 0. Exclude pairs with disjoint intervals; they are not interesting.
    /// 1. cvw < cwv => cuw <= cwu (if v before w, than so u)
    /// 2. cwu < cuw => cwv <= cvw (if u after w, than so v)
    /// 3. cuv <= cvu
    fn find_siblings(&self) -> VecB<Vec<NodeB>> {
        // For each node, the other nodes that must come before it.
        let mut must_come_before: VecB<Vec<NodeB>> = VecB::new(self.b);
        if !get_flag("siblings") {
            return must_come_before;
        }

        let intervals = self.intervals();
        let c = self.crossings().0;
        let mut siblings = 0;
        let mut rev_siblings = 0;
        let mut rev_sibling_weight = 0;
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
        info!("Found {siblings} siblings");
        info!("Surprises: {rev_siblings} of total score {rev_sibling_weight}");
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
