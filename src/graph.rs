use crate::node::*;
use std::{
    cmp::min,
    collections::{hash_map::Entry, HashMap},
    ops::{Index, IndexMut, Range},
};

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
}

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
        let mut score = 0;
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

pub fn one_sided_crossing_minimization(g: &Graph, bound: Option<u64>) -> Option<(Solution, u64)> {
    let initial_solution = initial_solution(g);
    let mut initial_score = g.score(&initial_solution);
    println!("Initial solution found, with score {initial_score}.");
    if let Some(bound) = bound {
        if bound < initial_score {
            initial_score = bound;
            println!("Set bound to {initial_score}.");
        }
    }
    let mut bb = Bb::new(g, initial_score);
    let sol = if bb.branch_and_bound() {
        Some((bb.best_solution, bb.best_score))
    } else {
        None
    };
    // Clear the \r line.
    eprintln!();
    eprintln!("B&B States    : {:>9}", bb.states);
    eprintln!("Unique subsets: {:>9}", bb.lower_bound_for_tail.len());
    sol
}

#[derive(Debug)]
pub struct Bb<'a> {
    pub g: &'a Graph,
    solution_len: usize,
    /// The first `solution_len` elements are fixed (the 'head').
    /// The remainder ('tail') is sorted in the initial order.
    solution: Solution,
    /// Partial score of the head.
    /// Includes:
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
    lower_bound_for_tail: HashMap<Vec<NodeB>, u64>,

    /// The number of states explored.
    states: u64,
}

impl<'a> Bb<'a> {
    pub fn new(g: &'a Graph, upper_bound: u64) -> Self {
        // Start with a greedy solution.
        let initial_solution = initial_solution(g);
        let initial_score = g.score(&initial_solution);

        let mut score = 0;
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
            score,
            upper_bound: min(upper_bound, initial_score),
            best_solution: initial_solution,
            best_score: initial_score,
            lower_bound_for_tail: HashMap::new(),
            states: 0,
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

        let tail = &self.solution[self.solution_len..];
        // TODO(ragnar): Figure out why tail isn't always sorted.
        let mut tail_copy = tail.to_vec();
        tail_copy.sort();

        if self.solution_len == self.solution.len() {
            debug_assert_eq!(self.score, self.g.score(&self.solution));
            let score = self.score;
            if score < self.upper_bound {
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
        // Additionally, if we already have a lower bound on how much the score of the tail exceeds the trivial lower bound, use that.
        let tail_excess = self
            .lower_bound_for_tail
            .get(&tail_copy)
            .copied()
            .unwrap_or_default();
        let my_lower_bound = self.score + tail_excess;

        // We can not find a solution of score < upper_bound.
        if self.upper_bound <= my_lower_bound {
            return false;
        }
        assert!(my_lower_bound <= self.best_score);

        let least_end = tail.iter().map(|u| self.g.intervals[*u].end).min().unwrap();

        let old_tail = tail.to_vec();
        let old_solution_len = self.solution_len;
        let old_score = self.score;

        let tail = &mut self.solution[self.solution_len..];

        // TODO(ragnar): Test whether re-optimizing the tail actually improves performance.
        // Results seem mixed.
        //
        // TODO(mees): use some heuristics to choose an ordering for trying nodes.
        // Sorting takes 20% of time and doesn't do much. Maybe add back later.
        // let get_median = |x| self.g.connections_b[x][self.g.connections_b[x].len() / 2];
        // tail.sort_by_key(|x| get_median(*x));
        // TODO: First check if we can swap the elements that were around the new leading element.
        commute_adjacent(self.g, tail);

        let mut solution = false;
        // If we skipped some children because of local pruning, do not update the lower bound for this tail.
        let mut skips = false;
        // Try each of the tail nodes as next node.
        for i in self.solution_len..self.solution.len() {
            // Swap the next tail node to the front of the tail.
            self.solution.swap(self.solution_len, i);

            let u = self.solution[self.solution_len];
            // Do not yet try vertices that start after some other vertex ends.
            // TODO: Think about the equality case.
            if self.g.intervals[u].start > least_end {
                continue;
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
            self.solution_len += 1;

            // Increment score for the new node, and decrement tail_lower_bound.
            self.score += self.g.partial_score(
                self.solution[self.solution_len - 1],
                &self.solution[self.solution_len..],
            );

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
                    tail_copy,
                    self.lower_bound_for_tail[&tail_copy],
                    self.states
                );
                assert_eq!(self.upper_bound, self.best_score);
                solution = true;
                // Early exit?
                if my_lower_bound == self.best_score {
                    // Restore the tail.
                    self.solution_len -= 1;
                    assert_eq!(self.solution_len, old_solution_len);
                    // self.solution[self.solution_len..=i].rotate_left(1);
                    self.solution[self.solution_len..].copy_from_slice(&old_tail);
                    return true;
                }
            }
            self.solution_len -= 1;
        }
        // Restore the tail.
        assert_eq!(self.solution_len, old_solution_len);
        // self.solution[self.solution_len..].rotate_left(1);
        self.solution[self.solution_len..].copy_from_slice(&old_tail);
        let tail_excess = self.upper_bound - old_score;
        if !skips {
            match self.lower_bound_for_tail.entry(tail_copy) {
                Entry::Occupied(mut e) => {
                    // We did a search without success so the value must grow.
                    assert!(tail_excess > *e.get());
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

#[cfg(test)]
mod test {
    use crate::generate::GraphType;

    use super::one_sided_crossing_minimization;

    /// Detects when we prune local states but then incorrectly update the lower bound.
    #[test]
    fn valid_lower_bound() {
        let n = 23;
        let extra = 14;
        let seed = 8546;
        eprintln!("{n} {extra} {seed}");
        let g = GraphType::Fan { n, extra }.generate(Some(seed));
        one_sided_crossing_minimization(&g, None);
    }
}
