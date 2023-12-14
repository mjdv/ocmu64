use crate::{graph::*, node::*};
use rand::{seq::SliceRandom, Rng, SeedableRng};

#[derive(clap::Parser)]
pub enum GraphType {
    /// A graph with 0 crossing, with some random extra edges.
    Fan { n: usize, extra: usize },
    /// A graph with star graphs with k edges each, and n nodes total.
    Star { n: usize, k: usize },
}

impl GraphType {
    pub fn generate(&self, seed: Option<u64>) -> Graph {
        let rng = &mut match seed {
            Some(seed) => rand_chacha::ChaCha8Rng::seed_from_u64(seed),
            None => rand_chacha::ChaCha8Rng::from_entropy(),
        };

        match *self {
            GraphType::Fan { n, extra } => fan_graph_with_random_edges(n, extra, rng),
            GraphType::Star { n, k } => stars(n, k, rng),
        }
    }
}

/// Add edges on either side at random, for total of n vertices.
// TODO: Different distributions of fan sizes.
pub fn fan_graph(n: usize, rng: &mut impl Rng) -> Graph {
    let mut g = Graph::default();
    let mut a = g.push_node_a();
    let mut b = g.push_node_b();
    g.push_edge(a, b);

    while a.0 + b.0 < n - 2 {
        if rng.gen_bool(0.5) {
            a = g.push_node_a();
        } else {
            b = g.push_node_b();
        }
        g.push_edge(a, b);
    }
    g
}

/// A fan graph with some random edges added in.
pub fn fan_graph_with_random_edges(n: usize, extra: usize, rng: &mut impl Rng) -> Graph {
    let mut g = fan_graph(n, rng);
    for _ in 0..extra {
        let a = NodeA(rng.gen_range(0..g.a.0));
        let b = NodeB(rng.gen_range(0..g.b.0));
        g.push_edge(a, b);
    }
    g
}

/// Create n/(k+1) star graphs rooted in B with endpoints in A.
/// Each of the B nodes has k random neighbours in A.
/// All A nodes have degree 1.
pub fn stars(n: usize, k: usize, rng: &mut impl Rng) -> Graph {
    let n = n / (k + 1);
    let a = NodeA(k * n);
    let b = NodeB(n);
    let mut g = Graph::with_sizes(a, b);
    // Shuffle the A nodes.
    let mut a_nodes = (NodeA(0)..a).collect::<Vec<_>>();
    a_nodes.shuffle(rng);
    // Each part of size k is a node of B.
    for b in NodeB(0)..b {
        for j in 0..k {
            let a = a_nodes[b.0 * k + j];
            g.push_edge(a, b);
        }
    }
    g
}
