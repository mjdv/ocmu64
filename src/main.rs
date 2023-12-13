#![feature(step_trait)]
pub mod generate;
pub mod graph;
pub mod node;

use std::env;

use crate::graph::one_sided_crossing_minimization;

fn main() {
    println!("Hello, world!");
    let args: Vec<String> = env::args().collect();
    let g = if args.len() == 1 {
        let mut g = graph::Graph::from_stdin()
            .expect("Did not get a graph in the correct format on stdin.");
        g.create_crossings();
        g
    } else {
        let file_path = &args[1];
        let mut g = graph::Graph::from_file(file_path).expect("Unable to read graph from file.");
        g.create_crossings();
        g
    };
    println!("Read a graph: ({:?}).", g);
    println!("Branch and bound...");
    let bb_output = one_sided_crossing_minimization(&g);
    if let Some((bb_solution, bb_score)) = bb_output {
        println!("Score of our beautiful solution: {bb_score}");
        println!("Our beautiful solution: {:?}", bb_solution);
    } else {
        println!("No solution found?!");
    }
    // println!("");
    // println!("Recursive...");
    // let (score, extension) = extend_solution_recursive(&mut sol);
    // println!("Score of our beautiful solution: {score}");
    // println!("Our beautiful solution: {:?}", extension);
}
