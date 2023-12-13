#![feature(step_trait)]

use std::env;

use crate::graph::{
    branch_and_bound, extend_solution_recursive, one_sided_crossing_minimization, NodeB, Solution,
};

mod graph;

fn main() {
    println!("Hello, world!");
    let args: Vec<String> = env::args().collect();
    let g = if args.len() == 1 {
        graph::Graph::from_stdin().expect("Did not get a graph in the correct format on stdin.")
    } else {
        let file_path = &args[1];
        graph::Graph::from_file(file_path).expect("Unable to read graph from file.")
    };
    println!("Read a graph: ({:?}).", g);
    let mut sol = graph::Solution {
        g: &g,
        solution: vec![],
    };

    for i in NodeB(0)..g.b {
        for j in NodeB(0)..g.b {
            for k in NodeB(0)..g.b {
                if g.crossings[i][j] > g.crossings[j][i]
                    && g.crossings[j][k] > g.crossings[k][j]
                    && g.crossings[k][i] > g.crossings[i][k]
                {
                    println!("Found triple ({i:?}, {j:?}, {k:?})");
                    println!(
                        "Degrees: ({} {} {})",
                        g.connections_b[i].len(),
                        g.connections_b[j].len(),
                        g.connections_b[k].len(),
                    );
                    println!(
                        "Adjacency lists: ({:?} {:?} {:?})",
                        g.connections_b[i], g.connections_b[j], g.connections_b[k],
                    );
                }
            }
        }
    }

    println!("Branch and bound...");
    let bb_output = one_sided_crossing_minimization(&g);
    if let Some((bb_solution, bb_score)) = bb_output {
        println!("Score of our beautiful solution: {bb_score}");
        println!("Our beautiful solution: {:?}", bb_solution.solution);
    } else {
        println!("No solution found?!");
    }
    // println!("");
    // println!("Recursive...");
    // let (score, extension) = extend_solution_recursive(&mut sol);
    // println!("Score of our beautiful solution: {score}");
    // println!("Our beautiful solution: {:?}", extension);
}
