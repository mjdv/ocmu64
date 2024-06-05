use std::ops::Range;

use colored::Colorize;
use log::info;

use crate::{get_flag, node::NodeA};

use super::{Graph, NodeB, Solution};

pub fn initial_solution(g: &Graph) -> Vec<NodeB> {
    info!("{}", "INITIAL SOLUTION".bold());
    let mut initial_solution = (NodeB(0)..g.b).collect::<Vec<_>>();
    sort_by_median(g, &mut initial_solution);
    commute_adjacent(g, &mut initial_solution);
    optimal_insert(g, &mut initial_solution);
    dp(g, &mut initial_solution);
    // optimal_insert(g, &mut initial_solution);
    sort_adjacent(g, &mut initial_solution);

    initial_solution
}

/// Sort adjacent nodes if they can be swapped without decreasing the score.
/// Useful for cleaner presentation.
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
fn optimal_insert(g: &Graph, sol: &mut Solution) {
    info!("before opt_insert {}", g.score(sol));
    let mut changed = true;
    let mut loops = 0;

    let mut prefix_max = g.prefix_max.v.clone();
    let mut suffix_min = g.suffix_min.v.clone();

    while changed {
        changed = false;

        // Move blocks of k at a time.
        for k in 1..3 {
            loops += 1;
            // Try to move i elsewhere.
            for i in 0..=sol.len() - k {
                let us = &sol[i..i + k];
                // let u = sol[i];
                // if g[u].is_empty() {
                //     continue;
                // }
                let ul = us.iter().map(|&u| *g[u].first().unwrap()).min().unwrap();
                let ur = us.iter().map(|&u| *g[u].last().unwrap()).max().unwrap();

                let mut best_delta = 0;
                let mut best_j = i;
                // move left
                let mut cur_delta = 0;
                for (j, &v) in sol[..i].iter().enumerate().rev() {
                    if prefix_max[j] < ul {
                        break;
                    }
                    for &u in us {
                        cur_delta -= g.cr(u, v);
                    }
                    if cur_delta > best_delta {
                        best_delta = cur_delta;
                        best_j = j;
                    }
                }
                // move right
                let mut cur_delta = 0;
                for (j, &v) in sol.iter().enumerate().skip(i + k) {
                    if suffix_min[j] > ur {
                        break;
                    }
                    for &u in us {
                        cur_delta += g.cr(u, v);
                    }
                    if cur_delta > best_delta {
                        best_delta = cur_delta;
                        best_j = j;
                    }
                }

                let mut update_range = |r: Range<usize>, sol: &mut [NodeB]| {
                    let len = sol.len();
                    if r.end == len {
                        suffix_min[len - 1] = *g[sol[len - 1]].first().unwrap_or(&g.a);
                    }
                    for j in (r.start..r.end.min(sol.len() - 1)).rev() {
                        suffix_min[j] = suffix_min[j + 1].min(*g[sol[j]].first().unwrap_or(&g.a));
                    }
                    if r.start == 0 {
                        prefix_max[0] = *g[sol[0]].last().unwrap_or(&NodeA(0));
                    }
                    for j in r.start.max(1)..r.end {
                        prefix_max[j] =
                            prefix_max[j - 1].max(*g[sol[j]].last().unwrap_or(&NodeA(0)));
                    }
                };

                // move left rotation
                if best_j < i {
                    sol[best_j..i + k].rotate_right(k);
                    changed = true;
                    update_range(best_j..i + k, sol);
                }
                // move right rotation
                if best_j > i {
                    sol[i..=best_j].rotate_left(k);
                    changed = true;
                    update_range(best_j..i + k, sol);
                }
            }
        }
    }
    info!("after opt_insert  {}", g.score(sol));
    info!("optimal insert loops {loops}");
}

/// k-wide DP along the main diagonal.
fn dp(g: &Graph, sol: &mut Vec<NodeB>) {
    if !get_flag("dp") {
        return;
    }
    let k = 15;

    // Indices:
    // - p: longest taken prefix
    // - mask: k next states. Note that mask[0] = 0.
    let mut dp = vec![vec![(isize::MAX, usize::MAX); 1 << k]; sol.len() + 1];
    dp[0][0] = (0, usize::MAX);

    for p in 0..=sol.len() {
        for mask in (0usize..1 << k.min(sol.len() - p)).step_by(2) {
            if p == 0 && mask == 0 {
                continue;
            }
            let mut best = isize::MAX;
            let mut best_u_idx = usize::MAX;
            // 0          63
            // 111.11110mask
            let full_mask = ((1 << 32) - 1) | (mask << 32);
            let high = 63 - full_mask.leading_zeros() as usize;
            // info!("{p} {mask:16b} {full_mask:48b} {high}");
            for j in (high - k + 1..=high).rev() {
                if (full_mask & (1 << j)) == 0 {
                    continue;
                }
                let new_full_mask = full_mask ^ (1 << j);
                let shift = new_full_mask.trailing_ones() as usize;
                let new_mask = new_full_mask >> shift;
                let new_p = p - 32 + shift;
                if new_p > sol.len() {
                    continue;
                }
                let base_score = dp[new_p][new_mask].0;
                //          v j=u=removed
                // 111..111000101000
                // 0               63
                // add score of 1s on the right.
                let u_idx = p + j - 32;
                if u_idx >= sol.len() {
                    continue;
                }
                let u = sol[u_idx];
                let mut delta = 0;
                for j2 in j + 1..=high {
                    if (new_full_mask & (1 << j2)) != 0 {
                        let v_idx = p + j2 - 32;
                        let v = sol[v_idx];
                        let cr = g.cr(u, v) as isize;
                        // info!("j2 {j2} v_idx {v_idx} v {v} delta += {cr}");
                        delta -= cr;
                    }
                }
                let new_score = base_score + delta;
                // info!("{p} {mask:16b} {full_mask:48b} {high} {j} {new_p} {new_mask:16b} {new_full_mask:48b} {shift} {base_score:6} {delta:6} {new_score:6} {u}");
                // assert!(delta >= 0);
                if new_score < best {
                    best = new_score;
                    best_u_idx = u_idx;
                }
            }
            // info!("dp[{p}][{mask:16b}] = ({best:8}, {best_u})");
            dp[p][mask] = (best, best_u_idx);
        }
    }

    let mut new_sol = vec![];
    let mut p = sol.len();
    let mut mask = 0;
    while p > 0 {
        let u_idx = dp[p][mask].1;
        new_sol.push(sol[u_idx]);
        let full_mask = ((1 << 32) - 1) | (mask << 32);
        let j = u_idx - p + 32;
        let new_full_mask = full_mask ^ (1 << j);
        let shift = new_full_mask.trailing_ones() as usize;
        let new_mask = new_full_mask >> shift;
        let new_p = p - 32 + shift;
        mask = new_mask;
        p = new_p;
    }
    new_sol.reverse();
    info!("before dp {}", g.score(sol));
    info!("after dp  {}", g.score(&mut new_sol));
    sol.copy_from_slice(&new_sol);
    info!("BEST DP IMPROVEMENT: {:?}", dp[sol.len()][0]);
}
