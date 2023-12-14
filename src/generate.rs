use std::iter::Step;

use rand::Rng;

use crate::{graph::*, node::*};

/// Add edges on either side at random, for total of n vertices.
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
