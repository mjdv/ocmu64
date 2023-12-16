use std::path::PathBuf;

use clap::Parser;
use ocmu64::{generate::GraphType, graph::*};

#[derive(clap::Parser)]
struct Args {
    /// Optional path to input file, or stdin by default.
    input: Option<PathBuf>,
    /// Optionally generate a graph instead of reading one.
    #[clap(subcommand)]
    generate: Option<GraphType>,
    #[clap(short, long)]
    seed: Option<u64>,
    #[clap(short, long)]
    upper_bound: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let g = match args.generate {
        Some(gt) => {
            assert!(args.input.is_none());
            gt.generate(args.seed)
        }
        None => match args.input {
            Some(f) => Graph::from_file(&f).expect("Unable to read graph from file."),
            None => {
                Graph::from_stdin().expect("Did not get a graph in the correct format on stdin.")
            }
        },
    };

    println!(
        "Read graph: {:?} {:?}, {} nodes, {} edges",
        g.a,
        g.b,
        g.a.0 + g.b.0,
        g.m
    );
    println!("Branch and bound...");
    let start = std::time::Instant::now();
    let bb_output = one_sided_crossing_minimization(&g, args.upper_bound);
    eprintln!("Branch and bound took {:?}", start.elapsed());
    if let Some((bb_solution, bb_score)) = bb_output {
        println!("Score of our beautiful solution: {bb_score}");
        if bb_solution.len() < 200 {
            println!("Our beautiful solution: {:?}", bb_solution);
        }
    } else {
        println!("No solution found?!");
    }
    // println!("");
    // println!("Recursive...");
    // let (score, extension) = extend_solution_recursive(&mut sol);
    // println!("Score of our beautiful solution: {score}");
    // println!("Our beautiful solution: {:?}", extension);
}
