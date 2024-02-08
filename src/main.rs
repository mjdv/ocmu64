use std::path::PathBuf;

use clap::Parser;
use itertools::Itertools;
use ocmu64::{generate::GraphType, graph::*, set_flags};

#[derive(clap::Parser)]
struct Args {
    /// Optionally generate a graph instead of reading one.
    #[clap(subcommand)]
    generate: Option<GraphType>,
    #[clap(global = true)]
    seed: Option<u64>,
    /// Optional path to input file, or stdin by default.
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

    let graphs = match (&args.generate, &args.input) {
        (Some(_), Some(_)) => panic!("Cannot generate and read a graph at the same time."),
        (Some(gt), None) => vec![(
            gt.generate(args.seed),
            format!("Generated with seed {:?}", args.seed),
        )],
        (None, Some(f)) => {
            // If f is a directory, iterate over all files in it.
            if f.is_dir() {
                let mut paths = std::fs::read_dir(f)
                    .unwrap()
                    .map(|x| x.unwrap().path())
                    .collect_vec();

                // TODO: human readable sort?
                paths.sort();
                paths
                    .iter()
                    .map(|path| {
                        eprintln!("Reading graph from file {:?}", path);
                        (
                            GraphBuilder::from_file(&path)
                                .expect("Unable to read graph from file."),
                            path.to_str().unwrap().to_string(),
                        )
                    })
                    .collect()
            } else {
                vec![(
                    GraphBuilder::from_file(&f).expect("Unable to read graph from file."),
                    f.to_str().unwrap().to_string(),
                )]
            }
        }
        (None, None) => {
            vec![(
                GraphBuilder::from_stdin()
                    .expect("Did not get a graph in the correct format on stdin."),
                "stdin".to_string(),
            )]
        }
    };

    for (g, p) in graphs {
        eprintln!("SOLVING GRAPH {p}");
        solve_graph(g, &args);
    }
}

fn solve_graph(g: GraphBuilder, args: &Args) {
    println!(
        "Read graph: {:?} {:?}, {} nodes, {} edges",
        g.a,
        g.b,
        g.a.0 + g.b.0,
        g.connections_a.iter().map(|x| x.len()).sum::<usize>()
    );
    // println!("Branch and bound...");
    let start = std::time::Instant::now();
    let bb_output = one_sided_crossing_minimization(g, args.upper_bound);
    // eprintln!("Branch and bound took {:?}", start.elapsed());
    if let Some((bb_solution, bb_score)) = bb_output {
        println!("Score of our beautiful solution: {bb_score}");
        // if bb_solution.len() < 200 {
        //     println!("Our beautiful solution: {:?}", bb_solution);
        // }
    } else {
        println!("No solution found?!");
    }
    // println!("");
    // println!("Recursive...");
    // let (score, extension) = extend_solution_recursive(&mut sol);
    // println!("Score of our beautiful solution: {score}");
    // println!("Our beautiful solution: {:?}", extension);
}
