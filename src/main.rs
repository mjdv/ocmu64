use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

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
    #[clap(long, global = true)]
    skip: bool,
    #[clap(global = true)]
    flags: Vec<String>,
}

fn main() {
    let args = Args::parse();

    set_flags(&args.flags);

    let mut graphs = match (&args.generate, &args.input) {
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

    // read database file using serde_json
    let db = Database::new(args.database.clone());

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum State {
        TriedBefore(u64),
        DoneBefore(u64),
        // Start time.
        Running(Instant),
        // Duration.
        Done(u64),
    }
    let state = graphs
        .iter()
        .map(|(_, p)| {
            (
                p.clone(),
                if db.get_score(p).is_some() {
                    State::DoneBefore(db.get_duration(p))
                } else {
                    State::TriedBefore(db.get_duration(p))
                },
            )
        })
        .collect::<Vec<_>>();

    fn duration_to_char(d: u64) -> &'static str {
        match d {
            0 => "0",
            ..=10 => "1",
            ..=100 => "2",
            ..=1000 => "3",
            _ => "4",
        }
    }

    fn update(state: &Mutex<Vec<(String, State)>>, p: &str, new_state: State) {
        let mut state = state.lock().unwrap();
        let idx = state.iter().position(|x| x.0 == p).unwrap();
        state[idx].1 = new_state;
        let summary = state
            .iter()
            .map(|x| {
                let (duration, color) = match x.1 {
                    State::DoneBefore(d) => (d, colored::Color::Magenta),
                    State::TriedBefore(d) => (d, colored::Color::Red),
                    State::Running(start) => (start.elapsed().as_secs(), colored::Color::Yellow),
                    State::Done(d) => (d, colored::Color::Green),
                };
                format!("{}", duration_to_char(duration).color(color))
            })
            .join("");
        let cnt = state
            .iter()
            .filter(|x| matches!(x.1, State::Done(_) | State::DoneBefore(_)))
            .count();
        let total = state.len();
        eprintln!("{cnt:>3}/{total:>3} {summary}");
    }

    let db = Arc::new(Mutex::new(db));
    let state = Arc::new(Mutex::new(state));

    // Sort graphs by unsolved first, by least amount of time tried.
    graphs.sort_by_key(|(_, p)| db.lock().unwrap().get_duration(p) as u64);

    {
        let db = db.clone();
        let state = state.clone();

        ctrlc::set_handler(move || {
            eprintln!("Caught Ctrl-C, saving database.");

            let mut db = db.lock().unwrap();

            // Update pending runs in database.
            let mut state = state.lock().unwrap();
            for (p, s) in state.iter_mut() {
                if let State::Running(start) = s {
                    let duration = start.elapsed().as_secs();
                    db.add_result(p, duration, None);
                    *s = State::Done(duration);
                }
            }

            db.save();
            std::process::exit(0);
        })
        .unwrap();
    }

    graphs.into_par_iter().for_each(|(g, p)| {
        if args.skip {
            if db.lock().unwrap().get_score(&p).is_some() {
                return;
            }
        }

        let start = std::time::Instant::now();
        update(&state, &p, State::Running(start));
        let score = solve_graph(g, &p, &args);
        let duration = start.elapsed().as_secs();
        db.lock().unwrap().add_result(&p, duration, score);
        update(&state, &p, State::Done(duration));
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
