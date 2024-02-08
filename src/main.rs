use std::{path::PathBuf, sync::Mutex, time::Duration};

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use ocmu64::{database::Database, generate::GraphType, graph::*, set_flags};
use rayon::prelude::*;

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
    #[clap(long, global = true, default_value = "db/exact.json")]
    database: PathBuf,
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

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum State {
        Pending,
        Running,
        Done(Duration),
    }
    let state = Mutex::new(vec![State::Pending; graphs.len()]);

    fn print(state: &Mutex<Vec<State>>) {
        let state = state.lock().unwrap();
        let summary = state
            .iter()
            .map(|&x| match x {
                State::Pending => format!("{}", "✗".red()),
                State::Running => format!("{}", "O".yellow()),
                State::Done(d) => {
                    let cnt = (d.as_secs() + 1).ilog10();
                    format!("{}", format!("{cnt}").green())
                }
            })
            .join("");
        let cnt = state
            .iter()
            .filter(|&&x| matches!(x, State::Done(_)))
            .count();
        let total = state.len();
        eprintln!("{cnt:>3}/{total:>3} {summary}");
    }

    // read database file using serde_json
    let db = Mutex::new(Database::new(args.database.clone()));

    graphs.into_par_iter().enumerate().for_each(|(i, (g, p))| {
        state.lock().unwrap()[i] = State::Running;
        print(&state);
        let start = std::time::Instant::now();
        let score = solve_graph(g, &p, &args);
        let duration = start.elapsed();
        state.lock().unwrap()[i] = State::Done(duration);
        print(&state);

        db.lock().unwrap().add_result(p, duration, score);
    });
}

fn solve_graph(g: GraphBuilder, p: &str, args: &Args) -> Option<u64> {
    eprintln!(
        "{p}: Read A={:?} B={:?}, {} nodes, {} edges",
        g.a,
        g.b,
        g.a.0 + g.b.0,
        g.connections_a.iter().map(|x| x.len()).sum::<usize>()
    );
    let output = one_sided_crossing_minimization(g, args.upper_bound);
    if let Some((_solution, score)) = output {
        eprintln!("{p}: SCORE: {score}");
        Some(score)
    } else {
        eprintln!("{p}: No solution found?!");
        None
    }
}
