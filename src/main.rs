use std::path::PathBuf;

use clap::Parser;
use ocmu64::{generate::GraphType, graph::*, set_flags};

#[derive(clap::Parser)]
struct Args {
    /// Optionally generate a graph instead of reading one.
    #[clap(subcommand)]
    generate: Option<GraphType>,
    /// Optional path to input file, or stdin by default.
    #[clap(global = true)]
    seed: Option<u64>,
    #[clap(short, long, global = true)]
    input: Option<PathBuf>,
    #[clap(short, long, global = true)]
    upper_bound: Option<u64>,
    #[clap(global = true)]
    flags: Vec<String>,
}

fn main() {
    let args = Args::parse();

    set_flags(&args.flags);

    let g = match args.generate {
        Some(gt) => {
            assert!(args.input.is_none());
            gt.generate(args.seed)
        }
        None => match args.input {
            Some(f) => GraphBuilder::from_file(&f).expect("Unable to read graph from file."),
            None => GraphBuilder::from_stdin()
                .expect("Did not get a graph in the correct format on stdin."),
        },
    };

    println!(
        "Read graph: {:?} {:?}, {} nodes, {} edges",
        g.a,
        g.b,
        g.a.0 + g.b.0,
        g.connections_a.iter().map(|x| x.len()).sum::<usize>()
    );
    println!("Branch and bound...");
    let start = std::time::Instant::now();
    let bb_output = one_sided_crossing_minimization(g, args.upper_bound);
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
