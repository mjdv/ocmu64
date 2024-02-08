use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
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
    #[clap(short, long, global = true)]
    timelimit: Option<u64>,
    #[clap(long, global = true)]
    subprocess: bool,
    #[clap(global = true)]
    flags: Vec<String>,
}

fn main() {
    let args = Args::parse();
    set_flags(&args.flags);

    set_limits(&args);

    if args.subprocess {
        main_subprocess(&args);
        return;
    }

    match (&args.generate, &args.input) {
        (Some(_), Some(_)) => panic!("Cannot generate and read a graph at the same time."),
        (Some(gt), None) => {
            let g = gt.generate(args.seed);
            solve_graph(g, &args);
        }
        (None, None) => {
            let g = GraphBuilder::from_stdin()
                .expect("Did not get a graph in the correct format on stdin.");
            solve_graph(g, &args);
        }
        (None, Some(path)) if path.is_file() => {
            // TODO: Write database for files.
            let g = GraphBuilder::from_file(path).expect("Unable to read graph from file.");
            solve_graph(g, &args);
        }
        (None, Some(path)) if path.is_dir() => {
            process_directory(path, &args);
        }
        _ => panic!(),
    };
}

/// Solve a single graph.
fn solve_graph(g: GraphBuilder, args: &Args) {
    eprintln!(
        "Read A={:?} B={:?}, {} nodes, {} edges",
        g.a,
        g.b,
        g.a.0 + g.b.0,
        g.connections_a.iter().map(|x| x.len()).sum::<usize>()
    );
    let output = one_sided_crossing_minimization(g, args.upper_bound);
    if let Some((_solution, score)) = output {
        eprintln!("SCORE: {score}");
    } else {
        eprintln!("No solution found?!");
    }
}

fn set_limits(args: &Args) {
    let set = |res, limit| {
        let rlimit = libc::rlimit {
            rlim_cur: limit as _,
            rlim_max: limit as _,
        };
        unsafe {
            libc::setrlimit(res, &rlimit);
        }
    };
    if let Some(time) = args.timelimit {
        set(libc::RLIMIT_CPU, time);
    }
}

/// When running as a subprocess, read a path and print the score to stdout as json.
fn main_subprocess(args: &Args) {
    let g = GraphBuilder::from_file(
        args.input
            .as_ref()
            .expect("Input path must be given for subprocess."),
    )
    .expect("Unable to read graph from file.");
    let output = one_sided_crossing_minimization(g, args.upper_bound);
    let score = output.map(|x| x.1);
    serde_json::to_writer(std::io::stdout(), &score).unwrap();
}

fn call_subprocess(path: &Path, args: &Args) -> Option<u64> {
    let mut arg = std::process::Command::new(std::env::current_exe().unwrap());
    arg.arg("--subprocess");
    arg.arg("--input").arg(path);
    arg.arg("--database").arg(&args.database);
    if let Some(time) = args.timelimit {
        arg.arg("--timelimit").arg(time.to_string());
    }
    arg.args(&args.flags);
    let output = arg.output().expect("Failed to execute subprocess.");
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice(&output.stdout).unwrap()
}

fn process_directory(dir: &Path, args: &Args) {
    let mut paths = std::fs::read_dir(dir)
        .unwrap()
        .map(|x| x.unwrap().path())
        .collect_vec();

    paths.sort();

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
        Failed(u64),
    }
    let state = paths
        .iter()
        .map(|p| {
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
            ..=3 => "1",
            ..=10 => "2",
            ..=30 => "3",
            ..=100 => "4",
            ..=300 => "5",
            ..=1000 => "6",
            ..=3000 => "7",
            _ => "8",
        }
    }

    fn print_state(state: &Vec<(PathBuf, State)>) {
        let summary = state
            .iter()
            .map(|x| {
                let (duration, color, style) = match x.1 {
                    State::DoneBefore(d) => (d, colored::Color::Magenta, false),
                    State::TriedBefore(d) => (d, colored::Color::Red, false),
                    State::Running(start) => {
                        (start.elapsed().as_secs(), colored::Color::Yellow, false)
                    }
                    State::Done(d) => (d, colored::Color::Green, false),
                    State::Failed(d) => (d, colored::Color::Red, true),
                };
                if style {
                    format!("{}", duration_to_char(duration).color(color).bold())
                } else {
                    format!("{}", duration_to_char(duration).color(color))
                }
            })
            .join("");
        let cnt = state
            .iter()
            .filter(|x| matches!(x.1, State::Done(_) | State::DoneBefore(_)))
            .count();
        let total = state.len();
        eprint!("{cnt:>3}/{total:>3} {summary}\r");
    }

    fn update(state: &Mutex<Vec<(PathBuf, State)>>, p: &Path, new_state: State) {
        let mut state = state.lock().unwrap();
        let idx = state.iter().position(|x| x.0 == p).unwrap();
        state[idx].1 = new_state;
    }

    let db = Arc::new(Mutex::new(db));
    let state = Arc::new(Mutex::new(state));

    // Sort graphs by unsolved first, by least amount of time tried.
    paths.sort_by_key(|p| db.lock().unwrap().get_duration(p) as u64);

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
                }
            }

            db.save();
            print_state(&state);
            std::process::exit(1);
        })
        .unwrap();
    }

    paths.into_iter().par_bridge().for_each(|p| {
        if args.skip {
            if db.lock().unwrap().get_score(&p).is_some() {
                return;
            }
        }

        let start = std::time::Instant::now();
        update(&state, &p, State::Running(start));
        print_state(&state.lock().unwrap());
        let score = call_subprocess(&p, &args);
        let duration = start.elapsed().as_secs();
        db.lock().unwrap().add_result(&p, duration, score);
        if score.is_some() {
            update(&state, &p, State::Done(duration));
        } else {
            update(&state, &p, State::Failed(duration));
        }
        // State will be printed after setting a new entry to Running.
    });
    print_state(&state.lock().unwrap());
    eprintln!();
}
