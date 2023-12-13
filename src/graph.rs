use std::{
    cmp::{max, min},
    fs::File,
    io::{stdin, BufRead, BufReader},
    iter::Step,
    ops::{Deref, Index, IndexMut},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct NodeA(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct NodeB(pub usize);

#[derive(Default, Debug)]
pub struct VecA<T> {
    v: Vec<T>,
}

#[derive(Default, Debug, Clone)]
pub struct VecB<T> {
    v: Vec<T>,
}

impl Step for NodeA {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Step::steps_between(&start.0, &end.0)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        usize::forward_checked(start.0, count).map(|x| NodeA(x))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Step::backward_checked(start.0, count).map(|x| NodeA(x))
    }
}

impl Step for NodeB {
    fn steps_between(start: &Self, end: &Self) -> Option<usize> {
        Step::steps_between(&start.0, &end.0)
    }

    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        usize::forward_checked(start.0, count).map(|x| NodeB(x))
    }

    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        Step::backward_checked(start.0, count).map(|x| NodeB(x))
    }
}

impl<T> Deref for VecA<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T> Deref for VecB<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<T> Index<NodeA> for VecA<T> {
    type Output = T;
    fn index(&self, index: NodeA) -> &Self::Output {
        &self.v[index.0]
    }
}

impl<T> IndexMut<NodeA> for VecA<T> {
    fn index_mut(&mut self, index: NodeA) -> &mut Self::Output {
        &mut self.v[index.0]
    }
}

impl<T> Index<NodeB> for VecB<T> {
    type Output = T;
    fn index(&self, index: NodeB) -> &Self::Output {
        &self.v[index.0]
    }
}

impl<T> IndexMut<NodeB> for VecB<T> {
    fn index_mut(&mut self, index: NodeB) -> &mut Self::Output {
        &mut self.v[index.0]
    }
}

#[derive(Debug)]
pub struct Graph {
    pub a: NodeA,
    pub b: NodeB,
    pub connections_a: VecA<Vec<NodeB>>,
    pub connections_b: VecB<Vec<NodeA>>,
    pub crossings: VecB<VecB<u64>>,
}

impl Graph {
    fn from_stream<T: BufRead>(stream: T) -> Result<Self, std::io::Error> {
        let mut a = NodeA::default();
        let mut b = NodeB::default();
        let mut connections_a: VecA<Vec<NodeB>> = VecA::default();
        let mut connections_b: VecB<Vec<NodeA>> = VecB::default();

        for line in stream.lines() {
            let line = line?;
            if line.starts_with('c') {
                continue;
            } else if line.starts_with('p') {
                let words = line.split(" ").collect::<Vec<&str>>();
                a = NodeA(words[2].parse().unwrap());
                b = NodeB(words[3].parse().unwrap());
                connections_a = VecA {
                    v: vec![vec![]; a.0],
                };
                connections_b = VecB {
                    v: vec![vec![]; b.0],
                };
            } else {
                let words = line.split(" ").collect::<Vec<&str>>();
                let x: NodeA = NodeA(words[0].parse::<usize>().unwrap() - 1);
                let y: NodeB = NodeB(words[1].parse::<usize>().unwrap() - a.0 - 1);
                connections_a[x].push(y);
                connections_b[y].push(x);
            }
        }

        for i in NodeA(0)..a {
            connections_a[i].sort();
        }

        for i in NodeB(0)..b {
            connections_b[i].sort();
        }

        let mut crossings: VecB<VecB<u64>> = VecB {
            v: vec![VecB { v: vec![0; b.0] }; b.0],
        };
        for node_i in NodeB(0)..b {
            for node_j in Step::forward(node_i, 1)..b {
                for edge_i in &connections_b[node_i] {
                    for edge_j in &connections_b[node_j] {
                        if edge_i > edge_j {
                            crossings[node_i][node_j] += 1;
                        } else if edge_i < edge_j {
                            crossings[node_j][node_i] += 1;
                        }
                    }
                }
            }
        }

        Ok(Self {
            a,
            b,
            connections_a,
            connections_b,
            crossings,
        })
    }

    /* Reads a graph from a file (in PACE format).
     */
    pub fn from_file(file_path: &str) -> Result<Self, std::io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        return Self::from_stream(reader);
    }

    /* Reads a graph from stdin (in PACE format).
     */
    pub fn from_stdin() -> Result<Self, std::io::Error> {
        return Self::from_stream(stdin().lock());
    }
}

// Crossings by having b1 before b2.
fn node_score(g: &Graph, b1: NodeB, b2: NodeB) -> u64 {
    g.crossings[b1][b2]
}

#[derive(Clone)]
pub struct Solution<'a> {
    pub g: &'a Graph,
    pub solution: Vec<NodeB>,
}

impl<'a> Solution<'a> {
    pub fn score(&self) -> u64 {
        let mut score = 0;
        for (j, &b2) in self.solution.iter().enumerate() {
            for &b1 in &self.solution[..j] {
                score += node_score(self.g, b1, b2);
            }
        }
        score
    }
}

pub fn extend_solution_recursive(solution: &mut Solution) -> (u64, Vec<NodeB>) {
    if solution.solution.len() == solution.g.b.0 {
        return (solution.score(), vec![]);
    }
    let mut best_score: u64 = std::u64::MAX;
    let mut best_extension: Vec<NodeB> = vec![];
    for new_node in 0..solution.g.b.0 {
        let new_node = NodeB(new_node);
        if !solution.solution.contains(&new_node) {
            solution.solution.push(new_node);
            let (new_score, new_extension) = extend_solution_recursive(solution);
            if new_score < best_score {
                best_score = new_score;
                best_extension = vec![new_node];
                best_extension.extend(new_extension);
            }
            solution.solution.pop();
        }
    }
    (best_score, best_extension)
}

fn commute_adjacent(g: &Graph, vec: &mut Vec<NodeB>) {
    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..vec.len() {
            if g.crossings[vec[i - 1]][vec[i]] > g.crossings[vec[i]][vec[i - 1]] {
                (vec[i - 1], vec[i]) = (vec[i], vec[i - 1]);
                changed = true;
            }
        }
    }
}

pub fn one_sided_crossing_minimization(g: &Graph) -> Option<(Solution, u64)> {
    let mut initial_solution = (NodeB(0)..g.b).collect::<Vec<NodeB>>();
    let get_median = |x| g.connections_b[x][g.connections_b[x].len() / 2];
    initial_solution.sort_by(|x, y| get_median(*x).cmp(&get_median(*y)));
    commute_adjacent(g, &mut initial_solution);
    let initial_score = Solution {
        g: g,
        solution: initial_solution,
    }
    .score();
    println!("Initial solution found, with score {initial_score}.");
    branch_and_bound(
        Solution {
            g: g,
            solution: vec![],
        },
        0,
        initial_score,
    )
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
pub fn branch_and_bound(
    partial_solution: Solution,
    mut lower_bound: u64,
    mut upper_bound: u64,
) -> Option<(Solution, u64)> {
    if partial_solution.solution.len() == partial_solution.g.b.0 {
        let score = partial_solution.score();
        return Some((partial_solution, score));
    }

    let mut remaining_nodes = vec![];
    for i in NodeB(0)..partial_solution.g.b {
        if !partial_solution.solution.contains(&i) {
            remaining_nodes.push(i);
        }
    }

    // To do: use some heuristics to look for better lower bounds.
    let mut my_lower_bound = partial_solution.score();
    for b1 in &partial_solution.solution {
        for b2 in &remaining_nodes {
            my_lower_bound += node_score(partial_solution.g, *b1, *b2);
        }
    }
    for (i2, b2) in remaining_nodes.iter().enumerate() {
        for b1 in &remaining_nodes[..i2] {
            my_lower_bound += min(
                node_score(partial_solution.g, *b1, *b2),
                node_score(partial_solution.g, *b2, *b1),
            );
        }
    }
    lower_bound = max(lower_bound, my_lower_bound);

    // No need to search this branch at all?
    if lower_bound >= upper_bound {
        return None;
    }

    // To do: use some heuristics to choose an ordering for trying nodes.
    let get_median =
        |x| partial_solution.g.connections_b[x][partial_solution.g.connections_b[x].len() / 2];
    remaining_nodes.sort_by(|x, y| get_median(*x).cmp(&get_median(*y)));
    commute_adjacent(partial_solution.g, &mut remaining_nodes);

    let mut best_solution = None;
    for new_node in remaining_nodes {
        if let Some(last_node) = partial_solution.solution.last() {
            if partial_solution.g.crossings[*last_node][new_node]
                > partial_solution.g.crossings[new_node][*last_node]
                && upper_bound < 90000
            {
                continue;
            }
        }
        let mut new_solution = partial_solution.clone();
        new_solution.solution.push(new_node);
        let extension = branch_and_bound(new_solution, lower_bound, upper_bound);
        if let Some((solution_candidate, candidate_score)) = extension {
            best_solution = Some((solution_candidate, candidate_score));
            upper_bound = candidate_score;
        }
        // Early exit?
        if lower_bound >= upper_bound {
            return best_solution;
        }
    }
    /*if best_solution.is_none() {
        return (None, lower_bound);
    }*/

    best_solution
}
