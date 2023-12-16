use super::*;
use std::{self, iter::Step, ops::Range};

#[derive(Debug, Default)]
pub struct GraphBuilder {
    pub a: NodeA,
    pub b: NodeB,
    pub connections_a: VecA<Vec<NodeB>>,
    pub connections_b: VecB<Vec<NodeA>>,
}

impl GraphBuilder {
    pub fn with_sizes(a: NodeA, b: NodeB) -> Self {
        Self {
            a,
            b,
            connections_a: VecA::new(a),
            connections_b: VecB::new(b),
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

    pub fn new(connections_a: VecA<Vec<NodeB>>, connections_b: VecB<Vec<NodeA>>) -> GraphBuilder {
        Self {
            a: connections_a.len(),
            b: connections_b.len(),
            connections_a,
            connections_b,
        }
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
        }
    }

    /// Permute the vertices of the graph so that B is roughly increasing.
    /// TODO: Store metadata to be able to invert the final solution.
    /// TODO: Allow customizing whether crossings are built.
    pub fn build(mut self) -> Graph {
        self.drop_singletons();
        self.sort_edges();
        self.permute(initial_solution(&self.to_graph()));
        self.sort_edges();
        self.to_graph()
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
    }

    /// Reconstruct `connections_a`, given `connections_b`.
    fn reconstruct_a(&mut self) {
        // Reconstruct A.
        self.connections_a = VecA::new(self.a);
        for b in NodeB(0)..self.b {
            for &a in self.connections_b[b].iter() {
                self.connections_a[a].push(b);
            }
        }
    }

    /// Reconstruct `connections_b`, given `connections_a`.
    fn reconstruct_b(&mut self) {
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

    fn crossings(&self) -> (VecB<VecB<u64>>, VecB<VecB<u64>>) {
        let mut crossings: VecB<VecB<u64>> = VecB {
            v: vec![VecB::new(self.b); self.b.0],
        };
        for node_i in NodeB(0)..self.b {
            for node_j in Step::forward(node_i, 1)..self.b {
                for edge_i in &self.connections_b[node_i] {
                    for edge_j in &self.connections_b[node_j] {
                        if edge_i > edge_j {
                            crossings[node_i][node_j] += 1;
                        }
                        if edge_i < edge_j {
                            crossings[node_j][node_i] += 1;
                        }
                    }
                }
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
