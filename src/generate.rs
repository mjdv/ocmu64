use std::{cmp::min, iter::Step};

use crate::{graph::*, node::*};
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_distr::{Distribution, Geometric, Uniform, WeightedIndex};

#[derive(clap::Parser, Debug)]
pub enum GraphType {
    /// A graph with 0 crossing, with some random extra edges.
    Fan { n: usize, extra: usize },
    /// A graph with star graphs with k edges each, and n nodes total.
    Star { n: usize, k: usize },
    /// A fan graph with some random modifications that keeps the number of
    /// crossings small. (The number of crossings parameter is an
    /// approximation.)
    LowCrossing {
        n: usize,
        crossings: u64,
        #[clap(default_value_t = 0.5)]
        p: f64,
    },
    LowCrossingClustered {
        n: usize,
        crossings: u64,
        #[clap(default_value_t = 0.5)]
        p: f64,
        #[clap(default_value_t = 100.0)]
        sigma: f64,
    },
}

impl GraphType {
    pub fn generate(&self, seed: Option<u64>) -> GraphBuilder {
        let rng = &mut match seed {
            Some(seed) => rand_chacha::ChaCha8Rng::seed_from_u64(seed),
            None => rand_chacha::ChaCha8Rng::from_entropy(),
        };

        match *self {
            GraphType::Fan { n, extra } => fan_graph_with_random_edges(n, extra, rng),
            GraphType::Star { n, k } => stars(n, k, rng),
            GraphType::LowCrossing { n, crossings, p } => low_crossing(n, crossings, p, rng),
            GraphType::LowCrossingClustered {
                n,
                crossings,
                p,
                sigma,
            } => low_crossing_clustered(n, crossings, p, sigma, rng),
        }
    }
}

/// Add edges on either side at random, for total of n vertices.
// TODO: Different distributions of fan sizes.
pub fn fan_graph(n: usize, rng: &mut impl Rng) -> GraphBuilder {
    let mut g = GraphBuilder::default();
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
pub fn fan_graph_with_random_edges(n: usize, extra: usize, rng: &mut impl Rng) -> GraphBuilder {
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
pub fn stars(n: usize, k: usize, rng: &mut impl Rng) -> GraphBuilder {
    let n = n / (k + 1);
    let a = NodeA(k * n);
    let b = NodeB(n);
    let mut g = GraphBuilder::with_sizes(a, b);
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

pub fn low_crossing_with_distribution<T: Distribution<usize>>(
    fan_graph: GraphBuilder,
    crossings: u64,
    b_distribution: T,
    p: f64,
    rng: &mut impl Rng,
) -> GraphBuilder {
    let mut g = fan_graph;
    let geometric_distribution: Geometric =
        Geometric::new(p).expect("Not a valid probability for geometric distribution.");

    let mut current_crossings: i64 = 0;
    while (current_crossings as u64) < crossings {
        let mut b: NodeB = NodeB(b_distribution.sample(rng));
        let mut a: NodeA = g[b][g[b].len() / 2];
        a = if rng.gen_bool(0.5) {
            NodeA(min(
                a.0 + (geometric_distribution.sample(rng) as usize + 1),
                g.a.0 - 1,
            ))
        } else {
            NodeA(a.0 - min(geometric_distribution.sample(rng) as usize + 1, a.0))
        };
        if g.try_push_edge(a, b) {
            let mut new_crossings: i64 = 0;
            for other_b in NodeB(0)..g.b {
                for other_a in &g[other_b] {
                    if (other_b < b && *other_a > a) || (other_b > b && *other_a < a) {
                        new_crossings += 1;
                    }
                }
            }
            while let Some(prev_b) = Step::backward_checked(b, 1) {
                let cpb = g.one_node_crossings(prev_b, b);
                let cbp = g.one_node_crossings(b, prev_b);
                if cpb <= cbp {
                    break;
                }
                new_crossings += cpb as i64;
                new_crossings -= cbp as i64;
                g.connections_b.swap(prev_b.0, b.0);
                b = prev_b;
            }
            while Step::forward(b, 1) < g.b {
                let next_b = Step::forward(b, 1);
                let cnb = g.one_node_crossings(next_b, b);
                let cbn = g.one_node_crossings(b, next_b);
                if cbn <= cnb {
                    break;
                }
                new_crossings += cnb as i64;
                new_crossings -= cbn as i64;
                g.connections_b.swap(b.0, next_b.0);
                b = next_b;
            }
            current_crossings += new_crossings;
        }
    }
    g.reconstruct_a();
    g
}

pub fn low_crossing(n: usize, crossings: u64, p: f64, rng: &mut impl Rng) -> GraphBuilder {
    let g = fan_graph(n, rng);
    let uniform_distribution = Uniform::try_from(0..g.b.0).unwrap();
    low_crossing_with_distribution(g, crossings, uniform_distribution, p, rng)
}

pub fn low_crossing_clustered(
    n: usize,
    crossings: u64,
    p: f64,
    sigma: f64,
    rng: &mut impl Rng,
) -> GraphBuilder {
    let g = fan_graph(n, rng);
    let weight_vector: Vec<f64> = (0..g.b.0)
        .map(|x| f64::exp(-((x as f64 - (g.b.0 / 2) as f64) / sigma).powi(2)) / sigma)
        .collect();
    println!("Sum = {}", weight_vector.iter().sum::<f64>());
    let normal_distribution = WeightedIndex::new(&weight_vector).unwrap();
    low_crossing_with_distribution(g, crossings, normal_distribution, p, rng)
}
