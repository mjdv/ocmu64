use std::path::PathBuf;

use clap::Parser;
use ocmu64::graph::*;

#[derive(clap::Parser)]
struct Args {
    /// Optional path to input file, or stdin by default.
    input: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let mut g = match args.input {
        Some(f) => Graph::from_file(&f).expect("Unable to read graph from file."),
        None => Graph::from_stdin().expect("Did not get a graph in the correct format on stdin."),
    };
    g.create_crossings();

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
