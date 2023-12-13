use std::{
    cmp::{max, min},
    fs::File,
    io::{stdin, BufRead, BufReader},
    iter::Step,
    ops::{Index, IndexMut},
};

use crate::node::{self, NodeA, NodeB};

#[derive(Debug)]
pub struct Graph {
    pub a: node::NodeA,
    pub b: node::NodeB,
    pub connections_a: node::VecA<Vec<node::NodeB>>,
    pub connections_b: node::VecB<Vec<node::NodeA>>,
    pub crossings: Option<node::VecB<node::VecB<u64>>>,
}

impl Graph {
    fn from_stream<T: BufRead>(stream: T) -> Result<Self, std::io::Error> {
        let mut a = node::NodeA::default();
        let mut b = node::NodeB::default();
        let mut connections_a: node::VecA<Vec<node::NodeB>> = node::VecA::default();
        let mut connections_b: node::VecB<Vec<node::NodeA>> = node::VecB::default();

        for line in stream.lines() {
            let line = line?;
            if line.starts_with('c') {
                continue;
            } else if line.starts_with('p') {
                let words = line.split(' ').collect::<Vec<&str>>();
                a = node::NodeA(words[2].parse().unwrap());
                b = node::NodeB(words[3].parse().unwrap());
                connections_a = node::VecA {
                    v: vec![vec![]; a.0],
                };
                connections_b = node::VecB {
                    v: vec![vec![]; b.0],
                };
            } else {
                let words = line.split(' ').collect::<Vec<&str>>();
                let x: node::NodeA = node::NodeA(words[0].parse::<usize>().unwrap() - 1);
                let y: node::NodeB = node::NodeB(words[1].parse::<usize>().unwrap() - a.0 - 1);
                connections_a[x].push(y);
                connections_b[y].push(x);
            }
        }

        for i in node::NodeA(0)..a {
            connections_a[i].sort();
        }

        for i in node::NodeB(0)..b {
            connections_b[i].sort();
        }

        Ok(Self {
            a,
            b,
            connections_a,
            connections_b,
            crossings: None,
        })
    }

    pub fn create_crossings(self: &mut Self) {
        let mut crossings: node::VecB<node::VecB<u64>> = node::VecB {
            v: vec![
                node::VecB {
                    v: vec![0; self.b.0]
                };
                self.b.0
            ],
        };
        for node_i in node::NodeB(0)..self.b {
            for node_j in Step::forward(node_i, 1)..self.b {
                for edge_i in &self.connections_b[node_i] {
                    for edge_j in &self.connections_b[node_j] {
                        if edge_i > edge_j {
                            crossings[node_i][node_j] += 1;
                        }
                        if edge_i < edge_j {
                            crossings[node_j][node_i] += 1;
                        }
                    }
                }
            }
        }
        self.crossings = Some(crossings);
    }

    /* Reads a graph from a file (in PACE format).
     */
    pub fn from_file(file_path: &str) -> Result<Self, std::io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        Self::from_stream(reader)
    }

    /* Reads a graph from stdin (in PACE format).
     */
    pub fn from_stdin() -> Result<Self, std::io::Error> {
        Self::from_stream(stdin().lock())
    }
}

// Crossings by having b1 before b2.
fn node_score(g: &Graph, b1: node::NodeB, b2: node::NodeB) -> u64 {
    if let Some(crossings) = &g.crossings {
        crossings[b1][b2]
    } else {
        let mut count = 0;
        for edge_i in &g.connections_b[b1] {
            for edge_j in &g.connections_b[b2] {
                if edge_i > edge_j {
                    count += 1;
                }
            }
        }
        count
    }
}

pub type Solution = Vec<node::NodeB>;

pub fn score(g: &Graph, solution: &Solution) -> u64 {
    let mut score = 0;
    for (j, &b2) in solution.iter().enumerate() {
        for &b1 in &solution[..j] {
            score += node_score(g, b1, b2);
        }
    }
    score
}

#[allow(unused)]
pub fn extend_solution_recursive(g: &Graph, solution: &mut Solution) -> (u64, Vec<node::NodeB>) {
    if solution.len() == g.b.0 {
        return (score(g, solution), vec![]);
    }
    let mut best_score: u64 = std::u64::MAX;
    let mut best_extension: Vec<node::NodeB> = vec![];
    for new_node in 0..g.b.0 {
        let new_node = node::NodeB(new_node);
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

fn commute_adjacent(g: &Graph, vec: &mut Vec<node::NodeB>) {
    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..vec.len() {
            if node_score(g, vec[i - 1], vec[i]) > node_score(g, vec[i], vec[i - 1]) {
                (vec[i - 1], vec[i]) = (vec[i], vec[i - 1]);
                changed = true;
            }
        }
    }
}

pub fn one_sided_crossing_minimization(g: &Graph) -> Option<(Solution, u64)> {
    let mut initial_solution = (node::NodeB(0)..g.b).collect::<Vec<node::NodeB>>();
    let get_median = |x: NodeB| g[x][g[x].len() / 2];
    initial_solution.sort_by_key(|x| get_median(*x));
    commute_adjacent(g, &mut initial_solution);
    let initial_score = score(g, &initial_solution);
    println!("Initial solution found, with score {initial_score}.");
    branch_and_bound(g, vec![], 0, initial_score)
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
    g: &Graph,
    partial_solution: Solution,
    mut lower_bound: u64,
    mut upper_bound: u64,
) -> Option<(Solution, u64)> {
    if partial_solution.len() == g.b.0 {
        let score = score(g, &partial_solution);
        return Some((partial_solution, score));
    }

    let mut remaining_nodes = vec![];
    for i in node::NodeB(0)..g.b {
        if !partial_solution.contains(&i) {
            remaining_nodes.push(i);
        }
    }

    // To do: use some heuristics to look for better lower bounds.
    let mut my_lower_bound = score(g, &partial_solution);
    for b1 in &partial_solution {
        for b2 in &remaining_nodes {
            my_lower_bound += node_score(g, *b1, *b2);
        }
    }
    for (i2, b2) in remaining_nodes.iter().enumerate() {
        for b1 in &remaining_nodes[..i2] {
            my_lower_bound += min(node_score(g, *b1, *b2), node_score(g, *b2, *b1));
        }
    }
    lower_bound = max(lower_bound, my_lower_bound);

    // No need to search this branch at all?
    if lower_bound >= upper_bound {
        return None;
    }

    // To do: use some heuristics to choose an ordering for trying nodes.
    let get_median = |x| g.connections_b[x][g.connections_b[x].len() / 2];
    remaining_nodes.sort_by_key(|x| get_median(*x));
    commute_adjacent(g, &mut remaining_nodes);

    let mut best_solution = None;
    for new_node in remaining_nodes {
        if let Some(last_node) = partial_solution.last() {
            if node_score(g, *last_node, new_node) > node_score(g, new_node, *last_node)
                && upper_bound < 90000
            {
                continue;
            }
        }
        let mut new_solution = partial_solution.clone();
        new_solution.push(new_node);
        let extension = branch_and_bound(g, new_solution, lower_bound, upper_bound);
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

impl Index<NodeA> for Graph {
    type Output = Vec<node::NodeB>;

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
    type Output = Vec<node::NodeA>;

    fn index(&self, index: NodeB) -> &Self::Output {
        &self.connections_b[index]
    }
}

impl IndexMut<NodeB> for Graph {
    fn index_mut(&mut self, index: NodeB) -> &mut Self::Output {
        &mut self.connections_b[index]
    }
}
