use colored::Colorize;
use log::info;

use crate::node::NodeA;

use super::{Graph, NodeB};

pub fn initial_solution(g: &Graph) -> Vec<NodeB> {
    info!("{}", "INITIAL SOLUTION".bold());
    let mut initial_solution = (NodeB(0)..g.b).collect::<Vec<_>>();
    sort_by_median(g, &mut initial_solution);
    commute_adjacent(g, &mut initial_solution);
    optimal_insert(g, &mut initial_solution);
    sort_adjacent(g, &mut initial_solution);

    initial_solution
}

fn sort_by_median(g: &Graph, solution: &mut [NodeB]) {
    let get_median = |x: NodeB| g[x][g[x].len() / 2];
    solution.sort_by_key(|x| get_median(*x));
}

/// Commute adjacent nodes as long as the score improves.
fn commute_adjacent(g: &Graph, vec: &mut [NodeB]) {
    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..vec.len() {
            if g.cr(vec[i - 1], vec[i]) > 0 {
                (vec[i - 1], vec[i]) = (vec[i], vec[i - 1]);
                changed = true;
            }
        }
    }
}

/// Keep iterating to find nodes that can be moved elsewhere.
fn optimal_insert(g: &Graph, sol: &mut [NodeB]) {
    let mut changed = true;
    let mut loops = 0;

    let mut prefix_max = g.prefix_max.v.clone();
    let mut suffix_min = g.suffix_min.v.clone();

    while changed {
        loops += 1;
        changed = false;
        // Try to move i elsewhere.
        for i in 0..sol.len() {
            let u = sol[i];
            if g[u].is_empty() {
                continue;
            }
            let ul = *g[u].first().unwrap();
            let ur = *g[u].last().unwrap();

            let mut best_delta = 0;
            let mut best_j = i;
            // move left
            let mut cur_delta = 0;
            for (j, &v) in sol[..i].iter().enumerate().rev() {
                if prefix_max[j] < ul {
                    break;
                }
                cur_delta -= g.cr(u, v);
                if cur_delta > best_delta {
                    best_delta = cur_delta;
                    best_j = j;
                }
            }
            // move right
            let mut cur_delta = 0;
            for (j, &v) in sol.iter().enumerate().skip(i + 1) {
                if suffix_min[j] > ur {
                    break;
                }
                cur_delta += g.cr(u, v);
                if cur_delta > best_delta {
                    best_delta = cur_delta;
                    best_j = j;
                }
            }
            if best_j > i {
                sol[i..=best_j].rotate_left(1);
                changed = true;
                if i == 0 {
                    prefix_max[0] = *g[sol[0]].last().unwrap_or(&NodeA(0));
                }
                for j in i.max(1)..=best_j {
                    prefix_max[j] = prefix_max[j - 1].max(*g[sol[j]].last().unwrap_or(&NodeA(0)));
                }
            }
            if best_j < i {
                sol[best_j..=i].rotate_right(1);
                changed = true;
                if i == sol.len() - 1 {
                    suffix_min[sol.len() - 1] = *g[sol[sol.len() - 1]].first().unwrap_or(&g.a);
                }
                for j in (best_j..=i.min(sol.len() - 2)).rev() {
                    suffix_min[j] = suffix_min[j + 1].min(*g[sol[j]].first().unwrap_or(&g.a));
                }
            }
        }
    }
    info!("optimal insert loops {loops}");
}

/// Sort adjacent nodes if they can be swapped without decreasing the score.
pub fn sort_adjacent(g: &Graph, sol: &mut [NodeB]) {
    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..sol.len() {
            if g.cr(sol[i - 1], sol[i]) >= 0 && sol[i - 1] > sol[i] {
                (sol[i - 1], sol[i]) = (sol[i], sol[i - 1]);
                changed = true;
            }
        }
    }
}
