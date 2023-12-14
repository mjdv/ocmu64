use std::iter::Step;

use rand::{seq::SliceRandom, Rng};

use crate::{graph::*, node::*};

/// Add edges on either side at random, for total of n vertices.
// TODO: Different distributions of fan sizes.
pub fn fan_graph(n: usize) -> Graph {
    let mut a = NodeA(0);
    let mut b = NodeB(0);
    let mut ca = VecA { v: vec![vec![b]] };
    let mut cb = VecB { v: vec![vec![a]] };
    let mut rng = rand::thread_rng();

    while ca.len().0 + cb.len().0 < n {
        if rng.gen_bool(0.5) {
            a = Step::forward(a, 1);
            ca.push(vec![]);
        } else {
            b = Step::forward(b, 1);
            cb.push(vec![]);
        }
        ca[a].push(b);
        cb[b].push(a);
    }

    Graph::new(ca, cb)
}

/// A fan graph with some random edges added in.
pub fn fan_graph_with_random_edges(n: usize, extra_edges: usize) -> Graph {
    let mut g = fan_graph(n);
    let mut rng = rand::thread_rng();
    for _ in 0..extra_edges {
        let a = NodeA(rng.gen_range(0..g.a.0));
        let b = NodeB(rng.gen_range(0..g.b.0));
        g[a].push(b);
        g[b].push(a);
    }
    g
}

/// Create n/(k+1) star graphs rooted in B with endpoints in A.
/// Each of the B nodes has k random neighbours in A.
/// All A nodes have degree 1.
pub fn stars(n: usize, k: usize) {
    let n = n / (k + 1);
    let a = NodeA(k * n);
    let b = NodeB(n);
    let mut ca = VecA {
        v: vec![vec![]; k * n],
    };
    let mut cb = VecB { v: vec![vec![]; n] };
    // Shuffle the A nodes.
    let mut a_nodes = (NodeA(0)..a).collect::<Vec<_>>();
    a_nodes.shuffle(&mut rand::thread_rng());
    // Each part of size k is a node of B.
    for b in NodeB(0)..b {
        for j in 0..k {
            let a = a_nodes[b.0 * k + j];
            ca[a].push(b);
            cb[b].push(a);
        }
    }
}
