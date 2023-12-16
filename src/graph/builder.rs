use super::*;
use std;

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

    // Permute the vertices of the graph so that B is roughly increasing.
    pub fn build(self) -> Graph {
        let mut g = Graph {
            a: self.connections_a.len(),
            b: self.connections_b.len(),
            m: self.connections_a.iter().map(|x| x.len()).sum::<usize>(),
            b_permutation: Default::default(),
            connections_a: self.connections_a,
            connections_b: self.connections_b,
            crossings: None,
            reduced_crossings: None,
        };

        for l in g.connections_a.iter_mut() {
            l.sort();
        }
        for l in g.connections_b.iter_mut() {
            l.sort();
        }
        let perm = VecB {
            v: initial_solution(&g),
        };
        let mut inv_perm = VecB {
            v: vec![NodeB(0); g.b.0],
        };
        for (i, &b) in perm.iter().enumerate() {
            inv_perm[b] = NodeB(i);
        }

        // A nbs are modified in-place.
        for b_nbs in g.connections_a.iter_mut() {
            for b in b_nbs {
                *b = inv_perm[*b];
            }
        }
        // B nbs are copied.
        let connections_b = VecB {
            v: perm
                .iter()
                .map(|&b| std::mem::take(&mut g.connections_b[b]))
                .collect(),
        };

        for l in g.connections_a.iter_mut() {
            l.sort();
        }

        Graph { connections_b, ..g }
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
