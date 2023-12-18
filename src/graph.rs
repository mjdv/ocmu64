use crate::{get_flag, node::*};
use std::{
    cmp::min,
    collections::{hash_map::Entry, HashMap},
    ops::{Deref, DerefMut, Index, IndexMut, Range},
};

use bitvec::prelude::*;
pub use builder::GraphBuilder;

mod builder;
mod io;

#[derive(Debug)]
#[non_exhaustive]
pub struct Graph {
    pub a: NodeA,
    pub b: NodeB,
    pub m: usize,
    pub connections_a: VecA<Vec<NodeB>>,
    pub connections_b: VecB<Vec<NodeA>>,
    pub b_permutation: VecB<NodeB>,
    pub crossings: Option<VecB<VecB<u64>>>,
    pub intervals: VecB<Range<NodeA>>,
    /// Stores max(cuv - cvu, 0).
    pub reduced_crossings: Option<VecB<VecB<u64>>>,
    pub self_crossings: u64,
    pub must_come_before: VecB<Vec<NodeB>>,
}

pub type Graphs = Vec<Graph>;

impl Graph {
    /// Crossings by having b1 before b2.
    fn node_score(&self, b1: NodeB, b2: NodeB) -> u64 {
        if let Some(crossings) = &self.crossings {
            crossings[b1][b2]
        } else {
            let mut count = 0;
            for edge_i in &self.connections_b[b1] {
                for edge_j in &self.connections_b[b2] {
                    if edge_i > edge_j {
                        count += 1;
                    }
                }
            }
            count
        }
    }

    /// The score of a solution.
    fn score(&self, solution: &[NodeB]) -> u64 {
        let mut score = self.self_crossings;
        for (j, &b2) in solution.iter().enumerate() {
            for &b1 in &solution[..j] {
                score += self.node_score(b1, b2);
            }
        }
        score
    }

    /// Compute the increase of score from fixing u before the tail.
    fn partial_score(&self, u: NodeB, tail: &[NodeB]) -> u64 {
        let rc = self
            .reduced_crossings
            .as_ref()
            .expect("Must have crossings.");
        let mut score = 0;
        for &v in tail {
            score += rc[u][v];
        }
        score
    }
}

pub type Solution = Vec<NodeB>;

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
            if g.node_score(vec[i - 1], vec[i]) > g.node_score(vec[i], vec[i - 1]) {
                (vec[i - 1], vec[i]) = (vec[i], vec[i - 1]);
                changed = true;
            }
        }
    }
}

fn initial_solution(g: &Graph) -> Vec<NodeB> {
    let mut initial_solution = (NodeB(0)..g.b).collect::<Vec<_>>();
    sort_by_median(g, &mut initial_solution);
    commute_adjacent(g, &mut initial_solution);
    initial_solution
}

fn oscm_part(g: &Graph, bound: Option<u64>) -> Option<(Solution, u64)> {
    let initial_solution = initial_solution(g);
    let mut initial_score = g.score(&initial_solution);
    eprintln!("Initial solution found, with score {initial_score}.");
    if let Some(bound) = bound {
        if bound < initial_score {
            initial_score = bound;
            eprintln!("Set bound to {initial_score}.");
        }
    }
    let mut bb = Bb::new(g, initial_score);
    if bb.branch_and_bound() {
        Some((bb.best_solution, bb.best_score))
    } else {
        Some((initial_solution, initial_score))
    }
}

pub fn one_sided_crossing_minimization(
    g: GraphBuilder,
    mut bound: Option<u64>,
) -> Option<(Vec<Solution>, u64)> {
    let mut score = g.self_crossings;
    let gs = g.build();
    eprintln!(
        "Part sizes: {:?}",
        gs.iter().map(|g| g.b.0).collect::<Vec<_>>()
    );

    let sol = 'sol: {
        let mut solutions = vec![];
        for (i, g) in gs.iter().enumerate() {
            let Some((part_sol, part_score)) = oscm_part(g, bound) else {
                eprintln!("No solution for part {i} of {}.", gs.len());
                break 'sol None;
            };
            score += part_score;
            solutions.push(part_sol);
            if let Some(bound) = bound.as_mut() {
                if *bound < part_score {
                    eprintln!("Ran out of bound at part {i} of {}.", gs.len());
                    eprintln!("{g:?}");
                    break 'sol None;
                }
                *bound -= part_score;
            }
        }
        Some((solutions, score))
    };

    // Clear the \r line.
    eprintln!();
    // eprintln!("Sols found    : {:>9}", bb.sols_found);
    // eprintln!("B&B States    : {:>9}", bb.states);
    // eprintln!("LB exceeded 1 : {:>9}", bb.lb_exceeded_1);
    // eprintln!("LB exceeded 2 : {:>9}", bb.lb_exceeded_2);
    // eprintln!("LB updates    : {:>9}", bb.lb_updates);
    // eprintln!("Unique subsets: {:>9}", bb.lower_bound_for_tail.len());
    // eprintln!("LB matching   : {:>9}", bb.lb_hit);
    sol
}

#[derive(Debug)]
pub struct Bb<'a> {
    pub g: &'a Graph,
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
    lower_bound_for_tail: HashMap<MyBitVec, u64>,

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
}

impl<'a> Bb<'a> {
    pub fn new(g: &'a Graph, upper_bound: u64) -> Self {
        // Start with a greedy solution.
        let initial_solution = initial_solution(g);
        let initial_score = g.score(&initial_solution);

        let mut score = g.self_crossings;
        let tail = &initial_solution;
        for (i2, &b2) in tail.iter().enumerate() {
            for &b1 in &tail[..i2] {
                score += min(g.node_score(b1, b2), g.node_score(b2, b1));
            }
        }

        Self {
            g,
            solution_len: 0,
            solution: initial_solution.clone(),
            tail_mask: MyBitVec::new(true, g.b.0),
            score,
            upper_bound: min(upper_bound, initial_score),
            best_solution: initial_solution,
            best_score: initial_score,
            lower_bound_for_tail: HashMap::default(),
            states: 0,
            sols_found: 0,
            lb_exceeded_1: 0,
            lb_exceeded_2: 0,
            lb_updates: 0,
            lb_hit: 0,
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

        // TODO(ragnar): Figure out why tail isn't always sorted.
        let tail = &self.solution[self.solution_len..];

        if self.solution_len == self.solution.len() {
            self.sols_found += 1;
            debug_assert_eq!(self.score, self.g.score(&self.solution));
            let score = self.score;
            if score < self.upper_bound {
                eprint!("Best score: {score:>9}\r");
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
        let tail_excess = self
            .lower_bound_for_tail
            .get(&self.tail_mask)
            .copied()
            .unwrap_or_default();
        let my_lower_bound = self.score + tail_excess;

        // We can not find a solution of score < upper_bound.
        if self.upper_bound <= my_lower_bound {
            self.lb_exceeded_2 += 1;
            return false;
        }
        assert!(my_lower_bound <= self.best_score);

        let least_end = tail.iter().map(|u| self.g.intervals[*u].end).min().unwrap();

        let old_tail = tail.to_vec();
        let old_solution_len = self.solution_len;
        let old_score = self.score;

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
        // If we skipped some children because of local pruning, do not update the lower bound for this tail.
        let mut skips = false;
        // Try each of the tail nodes as next node.
        'u: for i in self.solution_len..self.solution.len() {
            // Swap the next tail node to the front of the tail.
            self.solution.swap(self.solution_len, i);
            let u = self.solution[self.solution_len];

            // INTERVALS: Do not yet try vertices that start after some other vertex ends.
            // TODO: Think about the equality case.
            if self.g.intervals[u].start > least_end {
                continue;
            }

            // SIBLINGS: u must come after all the v listed here.
            // NOTE: It turns out this is not a valid optimization sadly.
            if get_flag("siblings") {
                for v in &self.g.must_come_before[u] {
                    if unsafe { *self.tail_mask.get_unchecked(v.0) } {
                        continue 'u;
                    }
                }
            }

            // NOTE: It's faster to not skip local inefficiencies, because then
            // we are guaranteed to have a valid lower bound on the tail that
            // can used for pruning.
            if false {
                // If this node commutes with the last one, fix their ordering.
                if let Some(&last) = self.solution.get(self.solution_len.wrapping_sub(1)) {
                    if self.g.node_score(last, u) > self.g.node_score(u, last)
                        // NOTE(ragnar): What is this check doing?
                        && self.upper_bound < 90000
                    {
                        skips = true;
                        continue;
                    }
                }
            }

            self.score = old_score;

            // Increment score for the new node, and decrement tail_lower_bound.
            // TODO: Early break once the upper_bound is hit.
            self.score += self.g.partial_score(
                self.solution[self.solution_len],
                &self.solution[self.solution_len + 1..],
            );

            self.solution_len += 1;
            debug_assert!(self.tail_mask[u.0]);
            unsafe { self.tail_mask.set_unchecked(u.0, false) };

            if self.branch_and_bound() {
                assert!(
                    my_lower_bound <= self.best_score,
                    "Found a solution with score {} but lower bound is {}
tail  {:?}
bound  {}
states {}
",
                    self.best_score,
                    my_lower_bound,
                    &self.solution[self.solution_len - 1..],
                    self.lower_bound_for_tail[&self.tail_mask],
                    self.states
                );
                assert_eq!(self.upper_bound, self.best_score);
                solution = true;
                // Early exit?
                if my_lower_bound == self.best_score {
                    // Restore the tail.
                    self.solution_len -= 1;
                    debug_assert!(!self.tail_mask[u.0]);
                    unsafe { self.tail_mask.set_unchecked(u.0, true) };
                    assert_eq!(self.solution_len, old_solution_len);
                    // self.solution[self.solution_len..=i].rotate_left(1);
                    self.solution[self.solution_len..].copy_from_slice(&old_tail);
                    self.lb_hit += 1;
                    return true;
                }
            }
            self.solution_len -= 1;
            debug_assert!(!self.tail_mask[u.0]);
            unsafe { self.tail_mask.set_unchecked(u.0, true) };
        }

        // Restore the tail.
        assert_eq!(self.solution_len, old_solution_len);
        debug_assert_eq!(self.tail_mask.count_zeros(), self.solution_len);
        // self.solution[self.solution_len..].rotate_left(1);
        self.solution[self.solution_len..].copy_from_slice(&old_tail);
        let tail_excess = self.upper_bound - old_score;
        if !skips {
            // TODO: Count updates and sets.
            match self.lower_bound_for_tail.entry(self.tail_mask.clone()) {
                Entry::Occupied(mut e) => {
                    // We did a search without success so the value must grow.
                    assert!(tail_excess > *e.get());
                    self.lb_updates += 1;
                    *e.get_mut() = tail_excess;
                }
                Entry::Vacant(e) => {
                    e.insert(tail_excess);
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
    use crate::{clear_flags, generate::GraphType, node::NodeB, set_flags};

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
        for (t, seed, drop) in [
            (GraphType::Star { n: 25, k: 4 }, 490, vec![]),
            (GraphType::Fan { n: 12, extra: 8 }, 7071, vec![]),
            (
                GraphType::Fan { n: 30, extra: 9 },
                7203,
                [0, 2, 5, 6, 7, 9, 10, 11].map(NodeB).to_vec(),
            ),
            (
                GraphType::LowCrossing {
                    n: 685,
                    crossings: 415,
                    p: 0.5,
                },
                2,
                vec![],
            ),
            (
                GraphType::LowCrossing {
                    n: 285,
                    crossings: 185,
                    p: 0.5,
                },
                46,
                vec![],
            ),
        ] {
            let mut g = t.generate(Some(seed));
            g.drop_b(&drop);
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
            println!("n = {}", n);
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
                        println!("{t:?} seed: {seed}");
                        // for u in NodeB(0)..g.b {
                        //     for v in &g.must_come_before[u] {
                        //         print!("{}<{} ", v.0, u.0);
                        //     }
                        // }
                        println!();
                        println!("score1: {}", score1);
                        println!("score2: {}", score2);
                        println!("sol1: {:?}", sol1);
                        println!("sol2: {:?}", sol2);
                        println!();
                        panic!();
                    }
                }
            }
        }
    }
}
