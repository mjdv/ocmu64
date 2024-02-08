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
        (None, Some(path)) => {
            let paths = if path.is_file() {
                vec![path.to_path_buf()]
            } else {
                std::fs::read_dir(path)
                    .unwrap()
                    .map(|x| x.unwrap().path())
                    .collect_vec()
            };
            if path
                .components()
                .find(|c| c.as_os_str() == "exact")
                .is_some()
            {
                process_exact_files(paths, &args);
            } else {
                for path in paths {
                    let g =
                        GraphBuilder::from_file(&path).expect("Unable to read graph from file.");
                    solve_graph(g, &args);
                }
            }
        }
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

fn process_exact_files(mut paths: Vec<PathBuf>, args: &Args) {
    // read database file using serde_json
    let db = Database::new("db/exact.json");
    let mut all_paths = std::fs::read_dir("input/exact")
        .unwrap()
        .map(|x| x.unwrap().path())
        .collect_vec();
    all_paths.sort();

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum OldState {
        Failed(u64),
        Solved(u64),
    }

    impl OldState {
        fn solved(&self) -> bool {
            matches!(self, OldState::Solved(_))
        }
        fn duration(&self) -> u64 {
            match self {
                OldState::Failed(d) => *d,
                OldState::Solved(d) => *d,
            }
        }
    }

    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    enum State {
        Pending,
        Running(Instant),
        Solved(u64),
        Failed(u64),
    }

    impl State {
        fn solved(&self) -> bool {
            matches!(self, State::Solved(_))
        }
        fn finished(&self) -> bool {
            matches!(self, State::Solved(_) | State::Failed(_))
        }
        fn duration(&self) -> Option<u64> {
            match self {
                State::Pending => None,
                State::Running(start) => Some(start.elapsed().as_secs()),
                State::Solved(d) => Some(*d),
                State::Failed(d) => Some(*d),
            }
        }
    }

    type States = Vec<(PathBuf, OldState, State)>;

    let state = all_paths
        .iter()
        .map(|p| {
            (
                p.clone(),
                if db.get_score(p).is_some() {
                    OldState::Solved(db.get_duration(p))
                } else {
                    OldState::Failed(db.get_duration(p))
                },
                State::Pending,
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

    fn print_old_state(state: &States) {
        let summary = state
            .iter()
            .map(|(_, so, _)| {
                let duration = so.duration();
                use colored::Color::*;
                let color = match so {
                    OldState::Failed(_) => Red,
                    OldState::Solved(_) => Green,
                };
                format!("{}", duration_to_char(duration).color(color))
            })
            .join("");
        let cnt = state.iter().filter(|x| x.1.solved()).count();
        let total = state.len();
        let ids_1 = (1..=state.len()).map(|x| format!("{}", x % 10)).join("");
        let ids_10 = (10..=state.len())
            .step_by(10)
            .map(|x| format!("{:>10}", x))
            .join("");
        eprintln!("{:>7} {}", "", ids_10);
        eprintln!("{:>7} {}", "", ids_1);
        eprintln!("{cnt:>3}/{total:>3} {summary}");
    }

    fn print_state(state: &States) {
        let summary = state
            .iter()
            .map(|(_, so, sn)| {
                let bold = sn.finished() && so.solved() != sn.solved();
                let underline = false; //matches!(sn, State::Running(_));
                let duration = sn.duration().unwrap_or(so.duration());
                use colored::Color::*;
                let color = match (so, sn) {
                    (OldState::Failed(_), State::Pending) => White,
                    (OldState::Failed(_), State::Running(_)) => Yellow,
                    (OldState::Failed(_), State::Solved(_)) => Green,
                    (OldState::Failed(_), State::Failed(_)) => Red,
                    (OldState::Solved(_), State::Pending) => White,
                    (OldState::Solved(_), State::Running(_)) => Yellow,
                    (OldState::Solved(_), State::Solved(_)) => Green,
                    (OldState::Solved(_), State::Failed(_)) => Magenta,
                };
                match (bold, underline) {
                    (true, true) => format!(
                        "{}",
                        duration_to_char(duration).color(color).bold().underline()
                    ),
                    (true, false) => format!("{}", duration_to_char(duration).color(color).bold()),
                    (false, true) => {
                        format!("{}", duration_to_char(duration).color(color).underline())
                    }
                    (false, false) => {
                        format!("{}", duration_to_char(duration).color(color))
                    }
                }
            })
            .join("");
        let cnt = state.iter().filter(|x| x.2.solved()).count();
        let total = state.len();
        eprint!("{cnt:>3}/{total:>3} {summary}\r");
    }

    fn update(state: &Mutex<Vec<(PathBuf, OldState, State)>>, p: &Path, new_state: State) {
        let mut state = state.lock().unwrap();
        let idx = state.iter().position(|x| x.0 == p).unwrap();
        state[idx].2 = new_state;
    }

    print_old_state(&state);

    let db = Arc::new(Mutex::new(db));
    let state = Arc::new(Mutex::new(state));

    // Sort graphs by unsolved first, by least amount of time tried.
    paths.sort_by_key(|p| db.lock().unwrap().get_duration(p) as u64);

    {
        let db = db.clone();
        let state = state.clone();

        ctrlc::set_handler(move || {
            let mut db = db.lock().unwrap();

            // Update pending runs in database.
            let mut state = state.lock().unwrap();
            for (p, _s_old, s_new) in state.iter_mut() {
                if let State::Running(start) = s_new {
                    let duration = start.elapsed().as_secs();
                    db.add_result(p, duration, None);
                    *s_new = State::Failed(duration);
                }
            }

            db.save();
            eprintln!();
            print_state(&state);
            eprintln!();
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
            update(&state, &p, State::Solved(duration));
        } else {
            update(&state, &p, State::Failed(duration));
        }
        // State will be printed after setting a new entry to Running.
    });
    print_state(&state.lock().unwrap());
    eprintln!();
}
