use crate::{
    get_flag,
    graph::builder::{is_practically_dominating_pair, is_practically_glued_pair, IsPDP},
    node::*,
    pattern_search::pattern_search,
};
use std::{
    cmp::min,
    collections::{hash_map::Entry, BTreeMap, HashMap},
    iter::Step,
    ops::{Deref, DerefMut, Index, IndexMut, Range},
};

use bitvec::prelude::*;
pub use builder::GraphBuilder;
use colored::Colorize;
use itertools::Itertools;
use log::*;

mod builder;
mod io;

/// Before[u][v] is true if u must come before v.
/// Before[u][u] is always false.
// TODO: Use bitvec instead?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BeforeType {
    Before,
    After,
    #[default]
    Unordered,
}
pub use BeforeType::*;
pub type Before = VecB<VecB<BeforeType>>;

/// Type for storing reduced crossings count.
/// i8 is not sufficient.
type CR = i32;

type ReducedCrossings = VecB<VecB<CR>>;

#[derive(Debug)]
#[non_exhaustive]
pub struct Graph {
    pub a: NodeA,
    pub b: NodeB,
    pub m: usize,
    pub connections_a: VecA<Vec<NodeB>>,
    pub connections_b: VecB<Vec<NodeA>>,
    pub b_permutation: VecB<NodeB>,
    /// Sum_{u,v} min(cuv, cvu)
    pub min_crossings: u64,
    /// cuv-cvu
    pub reduced_crossings: ReducedCrossings,
    pub intervals: VecB<Range<NodeA>>,
    pub self_crossings: u64,
    pub before: Before,
}

pub type Graphs = Vec<Graph>;

impl Graph {
    pub fn num_edges(&self) -> usize {
        self.connections_a.iter().map(|x| x.len()).sum()
    }

    /// c(u,v) - c(v,u)
    pub fn cr(&self, u: NodeB, v: NodeB) -> i64 {
        self.reduced_crossings[u][v] as _
    }

    /// The score of a solution.
    fn score(&self, solution: &Solution) -> u64 {
        assert_eq!(solution.len(), self.b.0, "Solution has wrong length.");
        let mut score = self.self_crossings + self.min_crossings;
        // HOT: 10% of parameterized time is here.
        // TODO: Compute rightmost edge of prefix and leftmost edge of suffix to
        // exclude pairs with cost 0.
        // TODO: Invert solution permutation and iterate over i,j and their positions in the solution, rather than over the solution order itself.
        for (j, &b1) in solution.iter().enumerate() {
            for &b2 in &solution[j + 1..] {
                score += self.cr(b1, b2).max(0) as u64;
            }
        }
        score
    }

    /// Compute the increase of score from fixing u before the tail.
    fn partial_score_2(&self, u: NodeB, tail: &[NodeB]) -> u64 {
        let mut score = 0;

        // HOT: 30% of B&B time is here.
        // TODO: Improve cache locality with the v index.
        // Maybe we can store which vertices have dropped from the range instead?
        // TODO: For each u, store the max v for which rc[u][v] is non-0, and assert that the tail is sorted by v.
        for &v in tail {
            score += self.cr(u, v).max(0) as u64;
        }
        score
    }
}

pub type Solution = Vec<NodeB>;

fn display_solution(g: &Graph, solution: &Solution, matrix: bool) -> String {
    let mut s = String::new();
    if log::log_enabled!(log::Level::Debug) {
        solution
            .chunk_by(|l, r| Step::forward(*l, 1) == *r)
            .for_each(|slice| {
                if slice.len() == 1 {
                    s.push_str(&format!("{} ", slice[0].0));
                } else {
                    s.push_str(&format!("{}-{} ", slice[0].0, slice.last().unwrap().0));
                }
            });
        s.push('\n');
    }

    if !matrix {
        return s;
    }

    for &u in solution {
        for &v in solution {
            let c = g.cr(u, v) as i64;
            let color = match c {
                ..=-1 => colored::Color::Red,
                0 => colored::Color::White,
                1.. => colored::Color::Green,
            };
            let forceduv = g.before[u][v] == Before;
            let forcedvu = g.before[u][v] == After;
            let c = if c.abs() < 10 {
                b'0' as i64 + c.abs()
            } else {
                (b'A' as i64 - 10 + c.abs()).min(b'Z' as i64)
            } as u8 as char;
            if forceduv {
                s.push_str(&format!(
                    "{}",
                    format!("{}", c).color(colored::Color::Black)
                ));
            } else if forcedvu {
                s.push_str(&format!(
                    "{}",
                    format!("{}", c).color(colored::Color::Black)
                ));
            } else {
                s.push_str(&format!("{}", format!("{}", c).color(color)));
            }
            // s.push_str(&format!(
            //     "{}",
            //     format!("{}", c).color(color).on_color(if forceduv {
            //         colored::Color::Red
            //     } else if forcedvu {
            //         colored::Color::Green
            //     } else {
            //         colored::Color::Black
            //     },)
            // ));
        }
        s.push('\n');
    }

    s
}

/// Naive recursive b! search.
#[allow(unused)]
pub fn extend_solution_recursive(g: &Graph, solution: &mut Solution) -> (u64, Vec<NodeB>) {
    if solution.len() == g.b.0 {
        return (g.score(solution), vec![]);
    }
    let mut best_score: u64 = u64::MAX;
    let mut best_extension: Vec<NodeB> = vec![];
    for new_node in NodeB(0)..g.b {
        if !solution.contains(&new_node) {
            solution.push(new_node);
            let (new_score, new_extension) = extend_solution_recursive(g, solution);
            if new_score < best_score {
                best_score = new_score;
                best_extension = vec![new_node];
                best_extension.extend(new_extension);
            }
            solution.pop();
        }
    }
    (best_score, best_extension)
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

/// Sort adjacent nodes if they can be swapped without decreasing the score.
fn sort_adjacent(g: &Graph, sol: &mut [NodeB]) {
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

/// Keep iterating to find nodes that can be moved elsewhere.
fn optimal_insert(g: &Graph, sol: &mut [NodeB]) {
    let mut changed = true;
    while changed {
        changed = false;
        // Try to move i elsewhere.
        for i in 0..sol.len() {
            let u = sol[i];
            let mut best_delta = 0;
            let mut best_j = i;
            // move left
            let mut cur_delta = 0;
            for (j, &v) in sol[..i].iter().enumerate().rev() {
                cur_delta -= g.cr(u, v);
                if cur_delta > best_delta {
                    best_delta = cur_delta;
                    best_j = j;
                }
            }
            // move right
            let mut cur_delta = 0;
            for (j, &v) in sol.iter().enumerate().skip(i + 1) {
                cur_delta -= g.cr(v, u);
                if cur_delta > best_delta {
                    best_delta = cur_delta;
                    best_j = j;
                }
            }
            if best_j > i {
                sol[i..=best_j].rotate_left(1);
                changed = true;
            }
            if best_j < i {
                sol[best_j..=i].rotate_right(1);
                changed = true;
            }
        }
    }
}

fn initial_solution(g: &Graph) -> Vec<NodeB> {
    let mut initial_solution = (NodeB(0)..g.b).collect::<Vec<_>>();
    sort_by_median(g, &mut initial_solution);
    commute_adjacent(g, &mut initial_solution);
    optimal_insert(g, &mut initial_solution);
    sort_adjacent(g, &mut initial_solution);

    initial_solution
}

fn oscm_part(g: &mut Graph, bound: Option<u64>) -> Option<(Solution, u64)> {
    let mut bb = Bb::new(g, bound);
    bb.branch_and_bound();

    info!("");
    info!("Sols found    : {:>9}", bb.sols_found);
    info!("B&B States    : {:>9}", bb.states);
    info!("LB exceeded 1 : {:>9}", bb.lb_exceeded_1);
    info!("LB exceeded 2 : {:>9}", bb.lb_exceeded_2);
    info!("LB updates    : {:>9}", bb.lb_updates);
    info!("Unique subsets: {:>9}", bb.tail_cache.len());
    info!("LB matching   : {:>9}", bb.lb_hit);
    info!("PDP yes       : {:>9}", bb.pdp_yes);
    info!("PDP no        : {:>9}", bb.pdp_no);
    info!("PDP comp. no  : {:>9}", bb.pdp_computed_no);
    info!("PDP cache yes : {:>9}", bb.pdp_cache_yes);
    info!("PDP cache no  : {:>9}", bb.pdp_cache_no);
    info!("PDP skip      : {:>9}", bb.pdp_skip);
    info!("glue yes      : {:>9}", bb.glue_yes);
    info!("glue no       : {:>9}", bb.glue_no);
    info!("glue no calls : {:>9}", bb.glue_no_calls);
    info!("tail update   : {:>9}", bb.tail_update);
    info!("tail insert   : {:>9}", bb.tail_insert);
    info!("tail skip     : {:>9}", bb.tail_skip);

    let best_score = bb.best_score;
    info!("Best score: {best_score}");
    let mut best_solution = bb.best_solution;
    debug_assert_eq!(g.score(&best_solution), best_score);

    if log::log_enabled!(log::Level::Debug) {
        sort_adjacent(g, &mut best_solution);
        debug_assert_eq!(g.score(&best_solution), best_score);
    }

    debug!(
        "Solution      : {}",
        display_solution(g, &best_solution, true)
    );
    if let Some(bound) = bound
        && best_score > bound
    {
        return None;
    }
    Some((best_solution, best_score))
}

pub fn one_sided_crossing_minimization(
    mut gb: GraphBuilder,
    mut bound: Option<u64>,
) -> Option<(Solution, u64)> {
    let g0 = gb.to_graph();
    let graph_builders = gb.build();
    let num_parts = graph_builders.len();
    let mut score = gb.self_crossings;

    let sol = 'sol: {
        let mut solution = Solution::default();
        for (i, mut gb) in graph_builders.into_iter().enumerate() {
            let mut g = gb.to_graph();
            let Some((part_sol, part_score)) = oscm_part(&mut g, bound) else {
                info!("No solution for part {i} of {num_parts}.");
                break 'sol None;
            };
            debug_assert_eq!(part_score, g.score(&part_sol), "WRONG SCORE FOR PART");
            // info!("{part_sol:?}");
            // Make sure that all `must_come_before` constraints are satisfied.
            for (u, v) in part_sol.iter().copied().tuple_combinations() {
                if g.before[u][v] == After {
                    info!("Part {i} of {num_parts} violates must_come_before constraints.");
                    info!("{u} {v}");
                    break 'sol None;
                }
            }

            pattern_search(&g, &part_sol);

            score += part_score;
            solution.extend(gb.invert(part_sol));
            if let Some(bound) = bound.as_mut() {
                if *bound < part_score {
                    info!("Ran out of bound at part {i} of {num_parts}.");
                    info!("{gb:?}");
                    break 'sol None;
                }
                *bound -= part_score;
            }
        }
        info!("{}", display_solution(&g0, &solution, false));
        assert_eq!(score, g0.score(&solution), "WRONG SCORE FOR FINAL SOLUTION");

        Some((solution, score))
    };
    sol
}

#[derive(Debug)]
pub struct Bb<'a> {
    pub g: &'a mut Graph,
    solution_len: usize,
    /// The first `solution_len` elements are fixed (the 'head').
    /// The remainder ('tail') is sorted in the initial order.
    solution: Solution,
    tail_mask: MyBitVec,
    /// Partial score of the head.
    /// Includes:
    /// - Self crossings from merged twins.
    /// - Crossings within the fixed prefix.
    /// - Crossing between the fixed prefix and the remainder.
    /// - The trivial sum of min(cuv, cvu) lower bound on the score of the tail.
    score: u64,
    /// We are looking for a solution with score strictly less than this.
    /// Typically the same as best_score, but may be lower if no good enough solution was found yet.
    upper_bound: u64,
    /// The best solution found so far.
    best_solution: Solution,
    /// The best solution score found so far.
    best_score: u64,

    /// This value is a lower bound on the score of the tail minus the trivial min(cuv,cvu) lower bound.
    /// TODO: Replace the key by a bitmask instead.
    /// The bitmask only has to be as wide as the cutwidth of the graph.
    /// It could be a template parameter to use the smallest width that is sufficiently large.
    /// TODO: Make the score a u32 here?
    ///
    /// Elements:
    /// - lower bound for score on tail on top of the trivial lower bound
    /// - a list of practically dominating pairs (v, u) for which v must come before u.
    /// - a list of u that are not practically dominated by any v.
    tail_cache: HashMap<MyBitVec, (u64, Vec<(NodeB, NodeB)>, Vec<NodeB>)>,

    /// The number of states explored.
    states: u64,
    /// The number of distinct solutions found.
    sols_found: u64,
    /// The number of times a lower bound was updated in the hashmap.
    lb_updates: u64,
    /// The number of times we return early because we found a solution with the same score as the lower bound.
    lb_hit: u64,
    lb_exceeded_1: u64,
    lb_exceeded_2: u64,
    pdp_yes: u64,
    pdp_no: u64,
    pdp_computed_no: u64,
    pdp_skip: u64,
    pdp_cache_no: u64,
    pdp_cache_yes: u64,
    glue_no: u64,
    glue_no_calls: u64,
    glue_yes: u64,
    tail_update: u64,
    tail_insert: u64,
    tail_skip: u64,
}

impl<'a> Bb<'a> {
    pub fn new(g: &'a mut Graph, upper_bound: Option<u64>) -> Self {
        // Start with a greedy solution.
        let initial_solution = (NodeB(0)..g.b).collect::<Vec<_>>();
        let initial_score = g.score(&initial_solution);

        info!("Initial solution found, with score {initial_score}.");

        let score = g.self_crossings + g.min_crossings;

        info!("Score lower bound: {score}");

        assert2::assert!(
            score <= initial_score,
            "Score lower bound is more than initial solution!"
        );

        Self {
            tail_mask: MyBitVec::new(true, g.b.0),
            g,
            solution_len: 0,
            solution: initial_solution.clone(),
            score,
            upper_bound: min(upper_bound.unwrap_or(u64::MAX), initial_score),
            best_solution: initial_solution,
            best_score: initial_score,
            tail_cache: HashMap::default(),
            states: 0,
            sols_found: 0,
            lb_exceeded_1: 0,
            lb_exceeded_2: 0,
            lb_updates: 0,
            lb_hit: 0,
            pdp_yes: 0,
            pdp_no: 0,
            pdp_computed_no: 0,
            pdp_skip: 0,
            pdp_cache_no: 0,
            pdp_cache_yes: 0,
            glue_no_calls: 0,
            glue_no: 0,
            glue_yes: 0,
            tail_update: 0,
            tail_insert: 0,
            tail_skip: 0,
        }
    }

    /// Branch and bound solution to compute OCM. The function will return either an
    /// optimal extension of the partial solution given, together with its score;
    /// or, it will return no solution, indicating that it is not possible to find
    /// an optimal solution to the global problem by extending this partial
    /// solution.
    ///
    /// # Arguments
    ///
    /// * `partial_solution` - A partial solution which we try to extend. If we
    /// output a solution at the end, it has to be an extension of this one.
    /// * `lower_bound` - This is an existing lower bound on `partial_solution`,
    /// that is, it can never be extended to a solution with less than `lower_bound`
    /// crossings. If you find an extension of `partial_solution` with `lower_bound`
    /// crossings, you can stop searching further: this solution is optimal.
    /// * `upper_bound` - A solution with this many crossings has already been
    /// found. Once you can prove that `partial_solution` cannot be extended to have
    /// fewer than `upper_bound` crossings, you can stop searching: this branch is
    /// not optimal.
    ///
    pub fn branch_and_bound(&mut self) -> bool {
        self.states += 1;

        if self.score >= self.upper_bound {
            self.lb_exceeded_1 += 1;
            return false;
        }

        debug_assert_eq!(self.tail_mask.count_zeros(), self.solution_len);

        let tail = &self.solution[self.solution_len..];

        debug_assert!(tail.is_sorted());

        if self.solution_len == self.solution.len() {
            self.sols_found += 1;
            debug_assert_eq!(self.score, self.g.score(&self.solution));
            let score = self.score;
            if score < self.upper_bound {
                if log::log_enabled!(log::Level::Info) {
                    eprint!("Best score: {score:>9}\r");
                }
                assert!(score < self.best_score);
                self.best_score = score;
                self.best_solution = self.solution.clone();
                // We found a solution of this score, so we are now looking for something strictly better.
                self.upper_bound = score;
                return true;
            } else if score < self.best_score {
                self.best_score = score;
                return false;
            } else {
                return false;
            }
        }

        // Compute a lower bound on the score as 3 parts of score:
        // 1. The true score of the head part.
        // 2. Crossings between the head and tail.
        // 3. A lower bound on the score of the tail.
        // Additionally, if we already have a lower bound on how much the score
        // of the tail exceeds the trivial lower bound, use that.
        // TODO: Find the largest suffix of `tail` that is a subset of tail_mask
        // and use that if no bound is available for `tail_mask`.
        let (tail_excess, mut local_before, mut not_dominated) = self
            .tail_cache
            .get(&self.tail_mask)
            // FIXME: Take the value instead of cloning it.
            .cloned()
            .unwrap_or((0, vec![], vec![]));
        let my_lower_bound = self.score + tail_excess;

        // We can not find a solution of score < upper_bound.
        if self.upper_bound <= my_lower_bound {
            self.lb_exceeded_2 += 1;
            return false;
        }
        assert!(my_lower_bound <= self.best_score);

        // HOT: ~20% of B&B time is here.
        let least_end = tail.iter().map(|u| self.g.intervals[*u].end).min().unwrap();

        let old_tail = tail.to_vec();
        let old_solution_len = self.solution_len;
        let old_score = self.score;

        // CAREFUL: Some optimizations may depend on the order of the tail!
        // TODO(ragnar): Test whether re-optimizing the tail actually improves performance.
        // Results seem mixed.
        //
        // TODO(mees): use some heuristics to choose an ordering for trying nodes.
        // Sorting takes 20% of time and doesn't do much. Maybe add back later.
        // let get_median = |x| self.g.connections_b[x][self.g.connections_b[x].len() / 2];
        // tail.sort_by_key(|x| get_median(*x));
        // TODO: First check if we can swap the elements that were around the new leading element.
        if false {
            let tail = &mut self.solution[self.solution_len..];
            commute_adjacent(self.g, tail);
        }

        let mut solution = false;

        let has_lpd = !get_flag("no_lpd"); // local practical dominating
        let has_lb = !get_flag("no_lb"); // local before
        let has_lb_cache = !get_flag("no_lb_cache"); // local before cache

        // Find new PDPs.
        let mut u_to_try = vec![];

        // Update before with local before.
        let mut existing_before = vec![];
        for &(v, u) in &local_before {
            if self.g.before[u][v] == After {
                existing_before.push((v, u));
            } else {
                self.g.before[v][u] = Before;
                self.g.before[u][v] = After;
            }
        }

        'u_to_try: {
            let tail = &self.solution[self.solution_len..];

            if self.solution_len > 0 && !get_flag("no_glue2") {
                'v: for i in self.solution_len..self.solution.len() {
                    let v = self.solution[i];
                    // early check in better ordered direction.
                    if self.g.cr(tail[0], v) < 0 {
                        self.glue_no_calls += 1;
                        continue 'v;
                    }
                    for &x in tail {
                        if self.g.cr(v, x) > 0 {
                            // x wants to be before v.
                            self.glue_no_calls += 1;
                            continue 'v;
                        }
                    }
                    // no x wants to be before v => glue uv.
                    u_to_try.push((i, v));
                    self.glue_yes += 1;
                    break 'u_to_try;
                }
                self.glue_no += 1;
            }

            // TODO: Why are there cases where uv are glued below but not already above.
            if self.solution_len > 0 && get_flag("glue") {
                let u = self.solution[self.solution_len - 1];
                for i in self.solution_len..self.solution.len() {
                    let v = self.solution[i];
                    if is_practically_glued_pair(
                        u,
                        v,
                        &self.g.before,
                        &self.g.reduced_crossings,
                        tail,
                    ) == IsPDP::Yes
                    {
                        u_to_try.push((i, v));
                        self.glue_yes += 1;
                        break 'u_to_try;
                    }
                    self.glue_no_calls += 1;
                }
                self.glue_no += 1;
            }

            'u: for i in self.solution_len..self.solution.len() {
                let u = self.solution[i];
                // eprintln!("Try {u} first");

                // INTERVALS: Do not yet try vertices that start after some other vertex ends.
                let u_range = &self.g.intervals[u];
                if u_range.start > least_end
                    || (u_range.start == least_end && u_range.end > least_end)
                {
                    continue;
                }

                // Skip u for which another v must come before.
                // HOT: ~20% of B&B time is here.
                for &v in tail {
                    if self.g.before[u][v] == After {
                        continue 'u;
                    }
                }

                // Skip u for which another v is newly PDP before u.
                if has_lpd {
                    if not_dominated.contains(&u) {
                        self.pdp_cache_no += 1;
                    } else {
                        let mut check_pdp = || {
                            for &v in tail {
                                match is_practically_dominating_pair(
                                    v,
                                    u,
                                    &self.g.before,
                                    &self.g.reduced_crossings,
                                    tail,
                                ) {
                                    builder::IsPDP::Skip => self.pdp_skip += 1,
                                    builder::IsPDP::No => self.pdp_no += 1,
                                    builder::IsPDP::Yes => {
                                        self.pdp_yes += 1;
                                        return Some(v);
                                    }
                                }
                            }
                            None
                        };

                        // Compute pdps for u below.
                        if let Some(v) = check_pdp() {
                            // eprintln!("Try {u} first => blocked by {v}");
                            assert!(self.g.before[v][u] != Before);
                            if has_lb {
                                local_before.push((v, u));
                                self.g.before[v][u] = Before;
                                self.g.before[u][v] = After;
                            }
                            continue 'u;
                        } else {
                            not_dominated.push(u);
                        }

                        self.pdp_computed_no += 1;
                    }
                }

                // eprintln!("Try {u} first => success");
                u_to_try.push((i, u));
            }
        }

        // If we skipped some children because of local pruning, do not update the lower bound for this tail.
        // Try each of the tail nodes as next node.
        let mut last_i = self.solution_len;
        'u: for (i, u) in u_to_try {
            // Swap the next tail node to the front of the tail.
            self.solution.swap(self.solution_len, i);
            // Make sure the tail remains sorted.
            if last_i + 1 <= i {
                self.solution[last_i + 1..=i].rotate_right(1);
            }
            last_i = i;

            debug_assert_eq!(self.solution[self.solution_len], u);
            debug_assert!(
                self.solution[(self.solution_len + 1).min(self.solution.len())..].is_sorted()
            );

            // NOTE: Adjust the score as if u was inserted in the optimal place in the prefix.
            let mut best_delta = 0;
            let mut best_i = self.solution_len;
            if !get_flag("no_optimal_insert") {
                let mut cur_delta = 0i64;
                for (i, v) in self.solution[..self.solution_len].iter().enumerate().rev() {
                    cur_delta -= self.g.cr(u, *v);
                    if cur_delta > best_delta as i64 {
                        best_delta = cur_delta as u64;
                        best_i = i;
                    }
                }
                if best_delta > 0 {
                    // warn!("best_delta: {}", best_delta);
                    // Rotate u into place.
                    self.solution[best_i..=self.solution_len].rotate_right(1);
                }
            }

            self.score = old_score;

            let my_lower_bound = my_lower_bound - best_delta;

            // Increment score for the new node, and decrement tail_lower_bound.
            // TODO: Early break once the upper_bound is hit.
            let partial_score = self
                .g
                .partial_score_2(u, &self.solution[self.solution_len + 1..]);

            self.score -= best_delta;
            self.score += partial_score;

            self.solution_len += 1;
            debug_assert!(self.tail_mask[u.0]);
            unsafe { self.tail_mask.set_unchecked(u.0, false) };

            if self.branch_and_bound() {
                // When moving u to the optimal insertion point, the optimal score can actually beat the known lower bound.
                // This is OK though; inserting points in order of optimal
                // solution will always append them at the back, and in that
                // situation the lower bound will be valid.
                if get_flag("no_optimal_insert") {
                    assert!(
                        my_lower_bound <= self.best_score,
                        "Found a solution with score {} but lower bound is {}
tail  {:?}
bound  {:?}
states {}
",
                        self.best_score,
                        my_lower_bound,
                        &self.solution[self.solution_len - 1..],
                        self.tail_cache.get(&self.tail_mask).map(|x| x.0),
                        self.states
                    );
                }
                assert_eq!(self.upper_bound, self.best_score);
                solution = true;
            }
            self.solution_len -= 1;
            debug_assert!(!self.tail_mask[u.0]);
            unsafe { self.tail_mask.set_unchecked(u.0, true) };
            // Rotate u back.
            self.solution[best_i..=self.solution_len].rotate_left(1);

            // Early exit?
            if my_lower_bound == self.best_score {
                self.lb_hit += 1;
                break 'u;
            }
        }

        // Clean up local_before.
        if has_lb {
            for &(v, u) in &local_before {
                assert!(self.g.before[v][u] == Before);
                self.g.before[v][u] = Unordered;
                self.g.before[u][v] = Unordered;
            }
            for &(v, u) in &existing_before {
                assert!(self.g.before[v][u] != Before);
                self.g.before[v][u] = Before;
                self.g.before[u][v] = After;
            }
        }

        // Restore the tail.
        assert_eq!(self.solution_len, old_solution_len);
        debug_assert_eq!(self.tail_mask.count_zeros(), self.solution_len);
        self.solution[self.solution_len..].copy_from_slice(&old_tail);

        assert!(old_score.min(self.best_score) <= self.upper_bound);
        // TODO: Can we bump the excess score to this value.
        let tail_excess = if solution {
            0
        } else {
            self.upper_bound.saturating_sub(old_score)
        };
        match self.tail_cache.entry(self.tail_mask.clone()) {
            Entry::Occupied(mut e) => {
                self.tail_update += 1;
                // We did a search without success so the value must grow.
                // assert!(tail_excess > *e.get());
                if tail_excess > e.get().0 {
                    e.get_mut().0 = tail_excess;
                    self.lb_updates += 1;
                }
                if has_lb_cache {
                    e.get_mut().1 = local_before;
                }
            }
            Entry::Vacant(e) => {
                self.tail_insert += 1;
                if has_lb_cache {
                    e.insert((tail_excess, local_before, not_dominated));
                } else {
                    e.insert((tail_excess, vec![], vec![]));
                }
            }
        }
        solution
    }
}

impl Index<NodeA> for Graph {
    type Output = Vec<NodeB>;

    fn index(&self, index: NodeA) -> &Self::Output {
        &self.connections_a[index]
    }
}

impl IndexMut<NodeA> for Graph {
    fn index_mut(&mut self, index: NodeA) -> &mut Self::Output {
        &mut self.connections_a[index]
    }
}

impl Index<NodeB> for Graph {
    type Output = Vec<NodeA>;

    fn index(&self, index: NodeB) -> &Self::Output {
        &self.connections_b[index]
    }
}

impl IndexMut<NodeB> for Graph {
    fn index_mut(&mut self, index: NodeB) -> &mut Self::Output {
        &mut self.connections_b[index]
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct MyBitVec(BitVec);
impl std::hash::Hash for MyBitVec {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        self.0.as_raw_slice().iter().for_each(|word| {
            word.hash(hasher);
        });
    }
}
impl MyBitVec {
    fn new(v: bool, n: usize) -> Self {
        let n = n.next_multiple_of(usize::BITS as _);
        MyBitVec(bitvec!(if v { 1 } else { 0 }; n))
    }
}
impl Deref for MyBitVec {
    type Target = BitVec;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for MyBitVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod test {
    use crate::{clear_flags, generate::GraphType, set_flags};

    use super::one_sided_crossing_minimization;

    /// Detects when we prune local states but then incorrectly update the lower bound.
    #[test]
    fn valid_lower_bound() {
        let n = 23;
        let extra = 14;
        let seed = 8546;
        eprintln!("{n} {extra} {seed}");
        let g = GraphType::Fan { n, extra }.generate(Some(seed));
        one_sided_crossing_minimization(g, None);
    }

    #[test]
    fn fuzz_fan() {
        for n in 10..100 {
            for extra in 0..50 {
                for seed in 0..10 {
                    eprintln!("{n} {extra} {seed}");
                    let g = GraphType::Fan { n, extra }.generate(Some(seed));
                    one_sided_crossing_minimization(g, None).expect("no solution found!");
                }
            }
        }
    }

    #[test]
    fn fuzz_star() {
        for n in 10..100 {
            for k in 1..9 {
                for seed in 0..100 {
                    eprintln!("{n} {k} {seed}");
                    let g = GraphType::Star { n, k }.generate(Some(seed));
                    one_sided_crossing_minimization(g, None).expect("no solution found!");
                }
            }
        }
    }

    #[test]
    fn fuzz_low_crossing() {
        for n in 10..100 {
            for crossings in 0..n as u64 {
                for seed in 0..10 {
                    for p in (1..9).map(|x| (x as f64) / 10.0) {
                        eprintln!("{n} {crossings} {seed} {p}");
                        let g = GraphType::LowCrossing { n, crossings, p }.generate(Some(seed));
                        one_sided_crossing_minimization(g, None).expect("no solution found!");
                    }
                }
            }
        }
    }

    #[test]
    fn bad_siblings() {
        let mut ok = true;
        for (t, seed) in [
            (GraphType::Star { n: 25, k: 4 }, 490),
            (GraphType::Fan { n: 12, extra: 8 }, 7071),
            (GraphType::Fan { n: 30, extra: 9 }, 7203),
            (
                GraphType::LowCrossing {
                    n: 685,
                    crossings: 415,
                    p: 0.5,
                },
                2,
            ),
            (
                GraphType::LowCrossing {
                    n: 285,
                    crossings: 185,
                    p: 0.5,
                },
                46,
            ),
        ] {
            let g = t.generate(Some(seed));
            clear_flags();
            let (sol1, score1) =
                one_sided_crossing_minimization(g.clone(), None).expect("no solution found!");
            set_flags(&["siblings"]);
            let (sol2, score2) =
                one_sided_crossing_minimization(g.clone(), None).expect("no solution found!");
            if score1 != score2 {
                eprintln!("DIFFERENCE FOUND!");
                eprintln!("{t:?} seed: {seed}");
                eprintln!("{g:?}");
                // for u in NodeB(0)..g.b {
                //     for v in &g.must_come_before[u] {
                //         eprint!("{}<{} ", v.0, u.0);
                //     }
                // }
                eprintln!();
                eprintln!("score1: {}", score1);
                eprintln!("score2: {}", score2);
                eprintln!("sol1: {:?}", sol1);
                eprintln!("sol2: {:?}", sol2);
                eprintln!();
                ok = false;
            }
        }
        assert!(ok);
    }

    #[ignore]
    #[test]
    fn fuzz_siblings() {
        for n in (5..10000).step_by(20) {
            eprintln!("n = {}", n);
            for k in (5..n).step_by(10) {
                for seed in 0..100 {
                    // let t = GraphType::Star { n, k };
                    // let t = GraphType::Fan { n, extra: k };
                    let t = GraphType::LowCrossing {
                        n,
                        crossings: k as _,
                        p: 0.5,
                    };
                    let g = t.generate(Some(seed));
                    clear_flags();
                    let (sol1, score1) = one_sided_crossing_minimization(g.clone(), None)
                        .expect("no solution found!");
                    set_flags(&["siblings"]);
                    let (sol2, score2) =
                        one_sided_crossing_minimization(g, None).expect("no solution found!");
                    if score1 != score2 {
                        eprintln!("{t:?} seed: {seed}");
                        // for u in NodeB(0)..g.b {
                        //     for v in &g.must_come_before[u] {
                        //         eprint!("{}<{} ", v.0, u.0);
                        //     }
                        // }
                        eprintln!();
                        eprintln!("score1: {}", score1);
                        eprintln!("score2: {}", score2);
                        eprintln!("sol1: {:?}", sol1);
                        eprintln!("sol2: {:?}", sol2);
                        eprintln!();
                        panic!();
                    }
                }
            }
        }
    }
}

pub fn draw(connections: &[Vec<NodeA>], sort: bool) {
    // Compress.
    let mut h: BTreeMap<NodeA, NodeA> = connections
        .iter()
        .flat_map(|x| x.iter().copied())
        .map(|x| (x, x))
        .collect();
    let mut a = NodeA(0);
    for x in h.values_mut() {
        *x = a;
        a = Step::forward(a, 1);
    }

    let mut connections: Vec<Vec<NodeA>> = connections
        .iter()
        .map(|x| x.iter().map(|x| *h.get(x).unwrap()).collect())
        .collect();

    let a = connections
        .iter()
        .map(|x| x.iter().map(|x| x.0).max().unwrap_or(0))
        .max()
        .unwrap_or(0)
        + 1;

    if sort {
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
    }

    let mut rows: Vec<Vec<Vec<NodeA>>> = vec![];
    for cs in connections.iter() {
        let first = cs.first().unwrap();
        let row = rows
            .iter_mut()
            .find(|r| r.last().is_some_and(|l| l.last().unwrap().0 + 1 < first.0));
        let row = match (sort, row) {
            (true, Some(row)) => row,
            _ => {
                rows.push(vec![]);
                rows.last_mut().unwrap()
            }
        };
        row.push(cs.clone());
    }
    debug!("Neighbours:");
    for row in rows {
        let mut line = vec![b' '; a];
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
