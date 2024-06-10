use crate::{
    get_flag,
    graph::builder::{is_practically_dominating_pair, is_practically_glued_pair, IsPDP},
    knapsack::KnapsackCache,
    max_with::MaxWith,
    node::*,
    pattern_search::pattern_search,
};
use std::{
    borrow::Borrow,
    cmp::{self, max, min},
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
pub mod initial_solution;
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
    /// cr = cuv-cvu
    pub reduced_crossings: ReducedCrossings,
    /// The range from leftmost negative element to just past the rightmost positive element.
    /// Typically, the prefix will all be positive because putting a random element before everything is expensive.
    /// Similarly, the suffix will all be negative because putting a random element after everything is expensive.
    pub cr_range: VecB<Range<NodeB>>,

    /// For each node, its first and last neighbor in A.
    pub intervals: VecB<Range<NodeA>>,

    /// The max neighbor in A for each prefix.
    pub prefix_max: VecB<NodeA>,
    /// The min neighbor in A for each suffix.
    pub suffix_min: VecB<NodeA>,

    pub self_crossings: u64,
    pub before: Before,

    /// Cache for knapsack-internal allocations.
    pub knapsack_cache: KnapsackCache,
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

        // Find the last position with edge intersecting u.
        // TODO: This can be improved by storing for each u the last positive cr[u][v].
        // TODO: Could store prefix sums and exclude vertices that are not in the tail.
        // TODO: Check for off-by-ones.
        let ur = self[u].last().unwrap();
        let idx = tail
            .binary_search_by(|x| self.suffix_min[*x].cmp(ur))
            .unwrap_or_else(|x| x);
        for &v in &tail[..idx] {
            score += self.cr(u, v).max(0) as u64;
        }
        score
    }
}

pub type Solution = Vec<NodeB>;

#[must_use]
fn display_solution(g: &Graph, solution: &mut Solution, matrix: bool) -> String {
    initial_solution::sort_adjacent(g, solution);
    let mut s = String::new();
    if log::log_enabled!(log::Level::Info) {
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

    for (i, &u) in solution.iter().enumerate() {
        let line = &mut vec![];
        s.push_str(&format!("{}\n", draw_edge(line, &g[u]).bold()));
        s.push_str(&format!("{i:4} "));
        for &v in &*solution {
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

fn oscm_part(g: &mut Graph, bound: Option<u64>) -> Option<(Solution, u64)> {
    let mut bb = Bb::new(g, bound);

    // If the initial solution already has optimal score, there is no need to B&B.
    // This mostly prevents cluttering the terminal output.
    if bb.score < bb.best_score {
        bb.branch_and_bound();

        info!("");
        info!("Sols found    : {:>9}", bb.sols_found);
        info!("B&B States    : {:>9}", bb.states);
        info!("states aft bst: {:>9}", bb.states_since_best_sol);
        info!("LB exceeded 1 : {:>9}", bb.lb_exceeded_1);
        info!("LB exceeded 2 : {:>9}", bb.lb_exceeded_2);
        info!("Reuse tail    : {:>9}", bb.reuse_tail);
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
        info!("tail insert   : {:>9}", bb.tail_insert);
        info!("tail update   : {:>9}", bb.tail_update);
        info!("tail excess up: {:>9}", bb.tail_excess_updates);
        info!("tail skip     : {:>9}", bb.tail_skip);
        info!("tail suffix   : {:>9}", bb.tail_suffix);
        info!("skip best     : {:>9}", bb.skip_best);
        info!("set best      : {:>9}", bb.set_best);

        let mut hist = bb.tail_excess_hist.iter().collect_vec();
        hist.sort();
        info!("Excess updates: {hist:?}\n");

        let mut hist = bb.tail_suffix_hist.iter().collect_vec();
        hist.sort();
        info!("Suffix pos    : {hist:?}\n");

        let mut hist = bb.u_poss.iter().collect_vec();
        hist.sort();
        info!("#u-to-try     : {hist:?}\n");
    }

    let best_score = bb.best_score;
    info!("Best score: {best_score}");
    let mut best_solution = bb.best_solution;
    if bb.implicit_solution {
        let mut mask = MyBitVec::new(true, bb.g.b.0);
        let mut tail = (NodeB(0)..bb.g.b).collect_vec();
        best_solution.clear();
        let mut prefix_max = NodeB(0);
        for _ in NodeB(0)..bb.g.b {
            let u = bb
                .tail_cache
                .get(&compress_tail_mask(&tail, &mask, prefix_max) as &dyn Key)
                .unwrap()
                .3
                .unwrap();
            prefix_max.max_with(u);
            best_solution.push(u);
            tail.remove(tail.iter().position(|x| *x == u).unwrap());
            mask.set(u.0, false);
        }
    }
    debug_assert_eq!(g.score(&best_solution), best_score);

    if log::log_enabled!(log::Level::Debug) {
        initial_solution::sort_adjacent(g, &mut best_solution);
        debug_assert_eq!(g.score(&best_solution), best_score);
    }

    debug!(
        "Solution      : {}",
        display_solution(g, &mut best_solution, true)
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
    // g0 is only used for verification and displaying.
    let g0 = if log::log_enabled!(log::Level::Debug) {
        Some(gb.to_graph(false))
    } else {
        None
    };
    let graph_builders = gb.build();
    let num_parts = graph_builders.len();
    let mut score = gb.self_crossings;

    let sol = 'sol: {
        let mut solution = Solution::default();
        for (i, mut gb) in graph_builders.into_iter().enumerate() {
            let (part_sol, part_score) = if gb.b == NodeB(1) {
                // For small parts, no need to do full B&B.
                (vec![NodeB(0)], 0)
            } else {
                info!("{}", "BUILD PART".bold());
                let mut g = gb.to_graph(true);
                let Some((mut part_sol, part_score)) = oscm_part(&mut g, bound) else {
                    info!("No solution for part {i} of {num_parts}.");
                    break 'sol None;
                };

                info!("Part score: {part_score}");
                info!("Part sol: {}", display_solution(&g, &mut part_sol, false));

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

                (part_sol, part_score)
            };

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
        if let Some(g0) = g0.as_ref() {
            // info!("{}", display_solution(g0, &mut solution, false));
            debug_assert_eq!(score, g0.score(&solution), "WRONG SCORE FOR FINAL SOLUTION");
        }

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
    /// For each prefix of the solution, the max node in A it connects to.
    solution_prefix_max: Vec<NodeA>,
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
    /// Indicates that the solution must be inferred from tail_cache.
    implicit_solution: bool,

    /// This value is a lower bound on the score of the tail minus the trivial min(cuv,cvu) lower bound.
    /// TODO: Replace the key by a bitmask instead.
    /// The bitmask only has to be as wide as the cutwidth of the graph.
    /// It could be a template parameter to use the smallest width that is sufficiently large.
    /// TODO: Make the score a u32 here?
    ///
    /// Elements:
    /// - lower bound for excess score on tail on top of the trivial lower bound
    /// - a list of practically dominating pairs (v, u) for which v must come before u.
    /// - a list of u that are not practically dominated by any v.
    /// - Option. When set, an optimal solution for the tail is known and this is the next node to choose.
    /// - Bool. True when the tail excess is exact, without crossing optimal inserts.
    tail_cache:
        HashMap<(usize, MyBitVec), (u64, Vec<(NodeB, NodeB)>, Vec<NodeB>, Option<NodeB>, bool)>,

    /// Highest vertex in prefix.
    prefix_max: NodeB,

    /// The number of states explored.
    states: u64,
    states_since_best_sol: u64,
    /// The number of distinct solutions found.
    sols_found: u64,
    /// The number of times we return early because we found a solution with the same score as the lower bound.
    lb_hit: u64,
    lb_exceeded_1: u64,
    lb_exceeded_2: u64,
    reuse_tail: u64,
    pdp_yes: u64,
    pdp_no: u64,
    pdp_computed_no: u64,
    pdp_skip: u64,
    pdp_cache_no: u64,
    pdp_cache_yes: u64,
    glue_no: u64,
    glue_no_calls: u64,
    glue_yes: u64,
    /// The number of times a lower bound was updated in the hashmap.
    tail_excess_updates: u64,
    tail_excess_hist: HashMap<u64, usize>,
    tail_update: u64,
    tail_insert: u64,
    tail_skip: u64,
    tail_suffix: u64,
    set_best: u64,
    skip_best: u64,
    tail_suffix_hist: HashMap<usize, usize>,
    u_poss: HashMap<usize, usize>,
}

#[derive(Copy, Clone)]
struct SmallKey<'a>(usize, &'a [usize]);

impl SmallKey<'_> {
    fn to_owned(&self) -> (usize, MyBitVec) {
        (self.0, MyBitVec(BitVec::from_slice(self.1)))
    }
}

trait Key {
    fn key(&self) -> (usize, &[usize]);
}

impl<'a> Eq for dyn Key + 'a {}

impl<'a> PartialEq for dyn Key + 'a {
    fn eq(&self, other: &dyn Key) -> bool {
        self.key() == other.key()
    }
}

impl<'a> std::hash::Hash for dyn Key + 'a {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let key = self.key();
        key.0.hash(state);
        key.1.iter().for_each(|word| {
            word.hash(state);
        });
    }
}

impl<'a> Key for SmallKey<'a> {
    fn key(&self) -> (usize, &[usize]) {
        (self.0, &self.1)
    }
}

impl<'a> Key for (usize, MyBitVec) {
    fn key(&self) -> (usize, &[usize]) {
        (self.0, self.1.as_raw_slice())
    }
}

impl<'a> Borrow<dyn Key + 'a> for (usize, MyBitVec) {
    fn borrow(&self) -> &(dyn Key + 'a) {
        self
    }
}

#[allow(unused)]
fn compress_tail(mut b: NodeB, mut tail: &[NodeB]) -> &[NodeB] {
    b = NodeB(b.0 - 1);
    while tail.len() > 1 && tail.last() == Some(&b) && tail[tail.len() - 2] == NodeB(b.0 - 1) {
        tail = tail.split_last().unwrap().1;
        b = NodeB(b.0 - 1);
    }
    tail
}

fn compress_tail_mask<'a>(
    tail: &[NodeB],
    tail_mask: &'a MyBitVec,
    prefix_max: NodeB,
) -> SmallKey<'a> {
    if tail.is_empty() {
        return SmallKey(tail_mask.len(), tail_mask.0.as_raw_slice());
    }
    let start = tail[0].0;
    let range = start / 64..(prefix_max.0 as usize + 1).div_ceil(64);
    // eprintln!("{start} {prefix_max} {range:?}");
    debug_assert!(prefix_max.0 + 1 >= start);
    let slice = &tail_mask.as_raw_slice()[range];
    // FIXME: This while loop can be made more efficient.
    // while slice.last() == Some(&usize::MAX) {
    //     slice = &slice[..slice.len() - 1];
    // }
    // Find last missing element of tail.
    // let start = tail[0].0 / 64;
    // let mut i = 0;
    // while i < tail.len() && tail_mask.len() - tail[i].0 != tail.len() - i {
    //     i += 1;
    // }
    // let end = (tail[i - 1].0 + 1).div_ceil(64).min(slice.len());

    SmallKey(start, slice)
}

impl<'a> Bb<'a> {
    pub fn new(g: &'a mut Graph, upper_bound: Option<u64>) -> Self {
        // Start with a greedy solution.
        let mut initial_solution = (NodeB(0)..g.b).collect::<Vec<_>>();
        let initial_score = g.score(&initial_solution);

        debug!(
            "Initial solution   : {}",
            display_solution(g, &mut initial_solution, true)
        );

        info!("Initial solution : {initial_score}");

        if let Some(upper_bound) = upper_bound
            && upper_bound < initial_score
        {
            info!("Upper bound      : {upper_bound}");
        }

        let score = g.self_crossings + g.min_crossings;

        info!("Score lower bound: {score}");

        assert2::assert!(
            score <= initial_score,
            "Score lower bound is more than initial solution!"
        );

        let b = g.b;
        Self {
            tail_mask: MyBitVec::new(true, b.0),
            g,
            solution_len: 0,
            solution: initial_solution.clone(),
            solution_prefix_max: vec![NodeA(0); b.0],
            score,
            upper_bound: min(upper_bound.unwrap_or(u64::MAX), initial_score),
            best_solution: initial_solution,
            best_score: initial_score,
            implicit_solution: false,
            tail_cache: HashMap::default(),

            prefix_max: NodeB(0),

            states: 0,
            states_since_best_sol: 0,
            sols_found: 0,
            lb_exceeded_1: 0,
            lb_exceeded_2: 0,
            reuse_tail: 0,
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
            tail_insert: 0,
            tail_update: 0,
            tail_excess_updates: 0,
            tail_excess_hist: HashMap::default(),
            tail_skip: 0,
            tail_suffix: 0,
            set_best: 0,
            skip_best: 0,
            tail_suffix_hist: HashMap::default(),
            u_poss: HashMap::default(),
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
    /// TODO: More input parameters, such as:
    /// - TODO: position / length of prefix.
    ///
    /// TODO: Move more arguments to here, such as best score in subtree.
    /// Returns:
    /// - whether a solution was found;
    /// - the leftmost optimal insert position.
    pub fn branch_and_bound(&mut self) -> (bool, usize) {
        self.states += 1;
        self.states_since_best_sol += 1;

        if self.score >= self.upper_bound {
            self.lb_exceeded_1 += 1;
            return (false, usize::MAX);
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
                self.states_since_best_sol = 0;
                self.best_score = score;
                self.best_solution.clone_from(&self.solution);
                self.implicit_solution = false;
                // We found a solution of this score, so we are now looking for something strictly better.
                self.upper_bound = score;
                return (true, usize::MAX);
            } else if score < self.best_score {
                self.best_score = score;
                self.states_since_best_sol = 0;
                return (false, usize::MAX);
            } else {
                return (false, usize::MAX);
            }
        }

        let old_prefix_max = self.prefix_max;

        // Compute a lower bound on the score as 3 parts of score:
        // 1. The true score of the head part.
        // 2. Crossings between the head and tail.
        // 3. A lower bound on the score of the tail.
        // Additionally, if we already have a lower bound on how much the score
        // of the tail exceeds the trivial lower bound, use that.

        // let mut original_excess = None;
        let (tail_excess, mut local_before, mut not_dominated) = match self
            .tail_cache
            .get(&compress_tail_mask(tail, &self.tail_mask, old_prefix_max) as &dyn Key)
        {
            Some(x) => {
                if x.3.is_some() && x.4 {
                    // We already have a solution for this tail.
                    let new_score = self.score + x.0;
                    self.implicit_solution = true;
                    if new_score < self.upper_bound {
                        self.states_since_best_sol = 0;
                        self.sols_found += 1;
                        self.reuse_tail += 1;
                        self.best_score = new_score;
                        self.upper_bound = new_score;
                        if log::log_enabled!(log::Level::Info) {
                            eprint!("Best score: {new_score:>9}\r");
                        }
                        // TODO: Update best solution.
                        return (true, usize::MAX);
                    } else if new_score < self.best_score {
                        self.states_since_best_sol = 0;
                        self.sols_found += 1;
                        self.reuse_tail += 1;
                        self.best_score = new_score;
                        return (false, usize::MAX);
                    }
                }
                *self.tail_suffix_hist.entry(0).or_default() += 1;
                // FIXME: Take the value instead of cloning it.
                (x.0, x.1.clone(), x.2.clone())
            }
            None => {
                if !get_flag("no_tail_suffix") {
                    let mut i = 0;
                    let tup = 'cleanup: {
                        for u in tail {
                            i += 1;
                            self.tail_suffix += 1;
                            unsafe { self.tail_mask.set_unchecked(u.0, false) };

                            let get = self.tail_cache.get(&compress_tail_mask(
                                &tail[i..],
                                &self.tail_mask,
                                old_prefix_max.max(*u),
                            )
                                as &dyn Key);
                            if let Some(get) = get {
                                *self.tail_suffix_hist.entry(i).or_default() += 1;
                                if get_flag("tail_suffix_full") {
                                    break 'cleanup (get.0, vec![], get.2.clone());
                                    // TODO: This is broken, but a similar idea may work.
                                    // break 'cache get.clone();
                                } else {
                                    break 'cleanup (get.0, vec![], vec![]);
                                }
                            }
                        }
                        (0, vec![], vec![])
                    };
                    for u in &tail[..i] {
                        unsafe { self.tail_mask.set_unchecked(u.0, true) };
                    }
                    tup
                } else {
                    (0, vec![], vec![])
                }
            }
        };

        let my_lower_bound = self.score + tail_excess;

        // We can not find a solution of score < upper_bound.
        if self.upper_bound <= my_lower_bound {
            self.lb_exceeded_2 += 1;
            return (false, usize::MAX);
        }
        assert!(my_lower_bound <= self.best_score);

        // HOT: ~20% of B&B time is here.
        let mut least_end = self.g.a;
        for v in tail {
            // If all remaining vertices start after the best end, we can stop iterating.
            if self.g.suffix_min[*v] >= least_end {
                break;
            }
            least_end = min(least_end, self.g.intervals[*v].end);
        }
        tail.iter().map(|u| self.g.intervals[*u].end).min().unwrap();

        let old_solution_len = self.solution_len;
        let old_score = self.score;

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

            // Only try vertices that start before the least end.
            // NOTE: Vertices who end at least_end exactly must be included.
            let idx_intersect_least_end = tail
                .binary_search_by(|x| {
                    if self.g.suffix_min[*x] <= least_end {
                        cmp::Ordering::Less
                    } else {
                        cmp::Ordering::Greater
                    }
                })
                .unwrap_or_else(|x| x);

            if self.solution_len > 0 && !get_flag("no_glue2") {
                // If there is a vertex v in the tail such that not a single u wants to go before it, then fix v.
                // FIXME: Smaller loop.
                'v: for i in self.solution_len..self.solution_len + idx_intersect_least_end {
                    let v = self.solution[i];
                    // early check in better ordered direction.
                    if self.g.cr(tail[0], v) < 0 {
                        self.glue_no_calls += 1;
                        continue 'v;
                    }

                    // TODO: Store last positive cr per v.
                    let vr = self.g.intervals[v].end;
                    let idx = tail
                        .binary_search_by(|x| self.g.suffix_min[*x].cmp(&vr))
                        .unwrap_or_else(|x| x);
                    for &x in &tail[..idx] {
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
                // FIXME: Smaller loop.
                for i in self.solution_len..self.solution_len + idx_intersect_least_end {
                    let v = self.solution[i];
                    // TODO: is this slow?
                    // TODO: Cache this.
                    if is_practically_glued_pair(
                        u,
                        v,
                        &self.g.before,
                        &self.g.reduced_crossings,
                        tail,
                        false, // FIXME: Can we make this strict?
                        &mut self.g.knapsack_cache,
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

            debug_assert!(not_dominated.is_sorted());

            'u: for i in self.solution_len..self.solution_len + idx_intersect_least_end {
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
                // Find the last v that intersects with u.
                let ur = self.g.intervals[u].end;
                let idx = tail
                    .binary_search_by(|x| self.g.suffix_min[*x].cmp(&ur))
                    .unwrap_or_else(|x| x);
                for &v in &tail[..idx] {
                    if self.g.before[u][v] == After {
                        continue 'u;
                    }
                }

                // Skip u for which another v is newly PDP before u.
                if has_lpd {
                    // Check if `not_dominated` contains `u`.
                    if not_dominated.binary_search(&u).is_ok() {
                        self.pdp_cache_no += 1;
                    } else {
                        let mut check_pdp = || {
                            // Test if there is a v in the tail that must come before u.
                            // Only test v that intersect u.
                            let ur = self.g.intervals[u].end;
                            let idx = tail
                                .binary_search_by(|x| self.g.suffix_min[*x].cmp(&ur))
                                .unwrap_or_else(|x| x);
                            for &v in &tail[..idx] {
                                match is_practically_dominating_pair(
                                    v,
                                    u,
                                    &self.g.before,
                                    &self.g.reduced_crossings,
                                    &tail[..idx],
                                    // FIXME: Can we allow cr[u][v]=0?
                                    false,
                                    &mut self.g.knapsack_cache,
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

        let mut leftmost_optimal_insert = usize::MAX;

        // If we skipped some children because of local pruning, do not update the lower bound for this tail.
        // Try each of the tail nodes as next node.
        let mut last_i = self.solution_len;
        let mut best_u = None;
        *self.u_poss.entry(u_to_try.len()).or_default() += 1;
        'u: for &(i, u) in &u_to_try {
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

            let mut u_leftmost_insert = self.solution_len;

            // NOTE: Adjust the score as if u was inserted in the optimal place in the prefix.
            let mut best_delta = 0;
            let mut best_i = self.solution_len;
            if !get_flag("no_optimal_insert") {
                let mut cur_delta = 0i64;
                let ul = self.g.intervals[u].start;
                // TODO: Check for off-by-ones.
                let idx = self.solution_prefix_max[..self.solution_len]
                    .binary_search(&ul)
                    .unwrap_or_else(|x| x);
                for (i, v) in self.solution[..self.solution_len]
                    .iter()
                    .enumerate()
                    .skip(idx)
                    .rev()
                {
                    cur_delta -= self.g.cr(u, *v);
                    if cur_delta > best_delta as i64 {
                        best_delta = cur_delta as u64;
                        best_i = i;
                    }
                }
                u_leftmost_insert.min_with(best_i);
                if best_delta > 0 {
                    // warn!("best_delta: {}", best_delta);
                    // Rotate u into place.
                    self.solution[best_i..=self.solution_len].rotate_right(1);
                }
            }

            self.prefix_max = max(old_prefix_max, u);
            // eprintln!("Set max to {old_prefix_max} and {u} -> {}", self.prefix_max);
            self.score = old_score;

            let my_lower_bound = my_lower_bound - best_delta;

            // Increment score for the new node, and decrement tail_lower_bound.
            // TODO: Early break once the upper_bound is hit.
            let partial_score = self
                .g
                .partial_score_2(u, &self.solution[self.solution_len + 1..]);

            self.score -= best_delta;
            self.score += partial_score;
            // Update prefix max.
            if self.solution_len == 0 {
                self.solution_prefix_max[self.solution_len] = self.g.intervals[u].end;
            } else {
                self.solution_prefix_max[self.solution_len] = self.g.intervals[u]
                    .end
                    .max(self.solution_prefix_max[self.solution_len - 1]);
            }

            self.solution_len += 1;
            debug_assert!(self.tail_mask[u.0]);
            unsafe { self.tail_mask.set_unchecked(u.0, false) };

            // eprintln!("RECURSE");
            let (sol_found, sub_lefmost_insert) = self.branch_and_bound();
            u_leftmost_insert.min_with(sub_lefmost_insert);

            if sol_found {
                // When moving u to the optimal insertion point, the optimal score can actually beat the known lower bound.
                // This is OK though; inserting points in order of optimal
                // solution will always append them at the back, and in that
                // situation the lower bound will be valid.
                if get_flag("no_optimal_insert") {
                    let tail = &self.solution[self.solution_len..];
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
                        self.tail_cache
                            .get(&compress_tail_mask(&tail, &self.tail_mask, old_prefix_max)
                                as &dyn Key)
                            .map(|x| x.0),
                        self.states
                    );
                }
                assert_eq!(self.upper_bound, self.best_score);
                solution = true;

                leftmost_optimal_insert = u_leftmost_insert;
                best_u = Some(u);

                // No 'crossing' optimal insert happened.
                if u_leftmost_insert >= self.solution_len - 1 {
                    self.set_best += 1;
                } else {
                    self.skip_best += 1;
                }
            }
            self.solution_len -= 1;
            debug_assert!(!self.tail_mask[u.0]);
            unsafe { self.tail_mask.set_unchecked(u.0, true) };
            // Rotate u back.
            self.solution[best_i..=self.solution_len].rotate_left(1);

            // NOTE: Not true because there may be additional optimal inserts not accounted for.
            // assert!(self.best_score >= my_lower_bound);

            // Early exit?
            if self.best_score == my_lower_bound {
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
        self.solution[self.solution_len..=last_i].rotate_left(1);
        let tail = &self.solution[self.solution_len..];
        // let ctail = compress_tail(self.g.b, tail).to_vec();

        assert!(old_score.min(self.best_score) <= self.upper_bound);

        // NOTE ON INTERACTION BETWEEN TAIL_EXCESS AND OPTIMAL INSERT.
        // - old_score is the score of prefix+trivial commute_2 bound on tail,
        //   i.e. full score apart from tail-tail intersection overhead.
        // - When no optimal inserts happen:
        //   - If a solution was found, the tail gets overhead=excess of best_score - old_score.
        //   - If no solution was found at cost *less than* upper_bound, the excess is upper_bound-old_score.
        // - When optimal inserts happen from the tail into the current prefix:
        //   - If a solution was found, best_score will be lower than the 'true'
        //     best_score for the tail, making the excess lower, which is fine.
        //   - If no solution was found with 'cheating', for sure no better
        //     solution will exist without optimal inserts.
        let crossing = leftmost_optimal_insert < self.solution_len;
        let tail_excess = if solution {
            if !crossing {
                assert!(old_score <= self.best_score);
            }
            self.best_score.saturating_sub(old_score)
        } else {
            if !crossing {
                assert!(old_score <= self.upper_bound);
            }
            self.upper_bound.saturating_sub(old_score)
        };

        // eprintln!("Excess {tail_excess} solution {solution} cur best {} upper bound {} old {old_score} best_u {best_u:?} sol len {} leftmost insert {leftmost_optimal_insert}\n for {ctail:?}", self.best_score, self.upper_bound,self.solution_len);

        // if let Some(new_score) = has_new_score {
        //     if leftmost_optimal_insert >= self.solution_len {
        //         assert_eq!(self.best_score, new_score, "Tail {ctail:?}");
        //     } else {
        //         assert!(self.best_score <= new_score, "Tail {ctail:?}");
        //     }
        //     assert!(solution);
        // }

        // NOTE ON INTERACTION BETWEEN OPTIMAL INSERT AND TAIL CACHING:
        // - The exact score of a tail can only be cached if no optimal insert crossed into the prefix.
        // - When a 'crossing' optimal insert did happen, the optimal move is not cached.
        match self
            .tail_cache
            .entry(compress_tail_mask(&tail, &self.tail_mask, old_prefix_max).to_owned())
        {
            Entry::Occupied(mut e) => {
                self.tail_update += 1;
                // We did a search without success so the value must grow.
                assert!(
                    solution || tail_excess > e.get().0,
                    "solution {solution} new excess {tail_excess} old excess {}",
                    e.get().0
                );

                if tail_excess > e.get().0 {
                    *self
                        .tail_excess_hist
                        .entry(tail_excess - e.get().0)
                        .or_default() += 1;
                    e.get_mut().0 = tail_excess;
                    self.tail_excess_updates += 1;
                }
                if has_lb_cache {
                    e.get_mut().1 = local_before;
                }
                if best_u.is_some() {
                    // FIXME: For some not understood reason, best_u can change over time.
                    // Must have to do with optimal_inset.
                    // assert!(
                    //     e.get().3.is_none() || e.get().3 == best_u,
                    //     "existing excess {} new excess {} original best {:?} new best {:?}",
                    //     e.get().0,
                    //     tail_excess,
                    //     e.get().3,
                    //     best_u
                    // );
                    e.get_mut().3 = best_u;
                    if crossing {
                        assert!(!e.get().4);
                    } else {
                        e.get_mut().4 = true;
                        if e.get_mut().4 {
                            assert_eq!(tail_excess, e.get().0);
                        }
                    }
                }
            }
            Entry::Vacant(e) => {
                self.tail_insert += 1;
                if has_lb_cache {
                    e.insert((
                        tail_excess,
                        local_before,
                        not_dominated,
                        best_u,
                        best_u.is_some() && !crossing,
                    ));
                } else {
                    e.insert((
                        tail_excess,
                        vec![],
                        vec![],
                        best_u,
                        best_u.is_some() && !crossing,
                    ));
                }
            }
        }

        (solution, leftmost_optimal_insert)
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

/// Wrapper type with better hash function.
#[derive(Debug, PartialEq, Eq, Clone)]
struct MyBitVec(BitVec);
impl std::hash::Hash for MyBitVec {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_raw_slice().iter().for_each(|word| {
            word.hash(state);
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
    if !get_flag("draw") {
        return;
    }
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
        let line = draw_row(&row);
        debug!("{}", line);
    }
}

fn draw_row(row: &Vec<Vec<Node<AT>>>) -> String {
    let mut line = vec![];
    for edge in row {
        draw_edge(&mut line, edge);
    }
    while line.last() == Some(&b' ') {
        line.pop();
    }
    let line = String::from_utf8(line).unwrap();
    line
}

fn draw_edge<'a>(line: &'a mut Vec<u8>, edge: &Vec<Node<AT>>) -> &'a str {
    let l = edge.last().unwrap().0;
    if line.len() <= l {
        line.resize(l + 1, b' ');
    }
    line[edge.first().unwrap().0..=edge.last().unwrap().0].fill(b'-');
    for a in edge {
        line[a.0] = match line[a.0] {
            b'-' => b'x',
            b'x' => b'2',
            c => c + 1,
        };
    }
    std::str::from_utf8(line).unwrap()
}
