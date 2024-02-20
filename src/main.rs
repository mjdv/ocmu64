use std::{
    iter::zip,
    path::{Path, PathBuf},
    process::{exit, Stdio},
    sync::{Arc, Mutex},
    time::Instant,
};

use clap::Parser;
use colored::Colorize;
use itertools::Itertools;
use log::info;
use ocmu64::{database::Database, generate::GraphType, graph::*, set_flags};
use rayon::prelude::*;

#[derive(clap::Parser)]
struct Args {
    /// Optionally generate a graph instead of reading one.
    #[clap(subcommand)]
    generate: Option<GraphType>,
    #[clap(short, long, global = true)]
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
    #[clap(short, long, global = true)]
    verbose: bool,
    #[clap(short, long, global = true)]
    print: bool,
    #[clap(long, global = true)]
    statsonly: bool,
    #[clap(global = true)]
    flags: Vec<String>,
}

/// Logging is only enabled when there is only a single testcase.
fn init_log(args: &Args, verbose: bool) {
    let verbosity = match (verbose, args.print) {
        (false, _) => 1,
        (true, false) => 2,
        (true, true) => 3,
    };
    stderrlog::new()
        .verbosity(verbosity)
        .show_level(false)
        .init()
        .unwrap();
}

fn main() {
    let args = &Args::parse();
    set_flags(&args.flags);

    if args.subprocess {
        init_log(args, args.verbose);
        main_subprocess(&args);
        return;
    }

    match (&args.generate, &args.input) {
        (Some(_), Some(_)) => panic!("Cannot generate and read a graph at the same time."),
        (Some(gt), None) => {
            init_log(args, true);
            let g = gt.generate(args.seed);
            solve_graph(g, &args);
        }
        (None, None) => {
            init_log(args, true);
            let g = GraphBuilder::from_stdin()
                .expect("Did not get a graph in the correct format on stdin.");
            solve_graph(g, &args);
        }
        (None, Some(path)) => {
            let mut paths = if path.is_file() {
                init_log(args, true);
                vec![path.to_path_buf()]
            } else {
                init_log(args, false);
                std::fs::read_dir(path)
                    .unwrap()
                    .map(|x| x.unwrap().path())
                    .collect_vec()
            };
            if args.statsonly {
                paths.sort();
                let graphs = paths.iter().map(|path| {
                    GraphBuilder::from_file(&path)
                        .expect("Unable to read graph from file.")
                        .build()
                });
                for (path, gs) in zip(&paths, graphs) {
                    let mut a = 0;
                    let mut b = 0;
                    let mut edges = 0;
                    for g in gs {
                        a += g.a.0;
                        b += g.b.0;
                        edges += g.num_edges();
                    }
                    eprintln!("{}: A={a:>5} B={b:>5} edges={edges:>6}", path.display());
                }

                return;
            }

            process_dir(paths, &args);
        }
    };
}

/// Solve a single graph.
fn solve_graph(g: GraphBuilder, args: &Args) -> Option<u64> {
    info!("Read A={:?} B={:?}, {} edges", g.a, g.b, g.num_edges());
    let output = one_sided_crossing_minimization(g, args.upper_bound);
    let score = output.map(|x| x.1);
    if let Some(score) = score {
        info!("SCORE: {score}");
    } else {
        info!("No solution found?!");
    }
    score
}

/// When running as a subprocess, read a path and print the score to stdout as json.
fn main_subprocess(args: &Args) {
    let time = args.timelimit.unwrap();
    unsafe {
        libc::setrlimit(
            libc::RLIMIT_CPU,
            &libc::rlimit {
                rlim_cur: time,
                rlim_max: time,
            },
        );
    }

    let g = GraphBuilder::from_file(
        args.input
            .as_ref()
            .expect("Input path must be given for subprocess."),
    )
    .expect("Unable to read graph from file.");
    let score = solve_graph(g, args);
    serde_json::to_writer(std::io::stdout(), &score).unwrap();
}

/// If a timelimit is set, run in a subprocess.
fn call_subprocess(path: &Path, args: &Args) -> Option<u64> {
    let Some(time) = args.timelimit else {
        // When no timelimit is set, just run directly in the same process.
        let g = GraphBuilder::from_file(path).expect("Unable to read graph from file.");
        return solve_graph(g, args);
    };

    let mut arg = std::process::Command::new(std::env::current_exe().unwrap());
    arg.arg("--subprocess");
    arg.arg("--input").arg(path);
    arg.arg("--timelimit").arg(time.to_string());
    if let Some(ub) = args.upper_bound {
        arg.arg("--upper-bound").arg(ub.to_string());
    }
    if log::log_enabled!(log::Level::Info) {
        arg.arg("--verbose");
    }
    arg.args(&args.flags);
    arg.stderr(Stdio::inherit());
    let output = arg.output().expect("Failed to execute subprocess.");
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice(&output.stdout).unwrap()
}

fn process_dir(mut paths: Vec<PathBuf>, args: &Args) -> Option<()> {
    let dir = paths.first()?.parent()?;
    let dirname = dir.file_name()?.to_str()?;

    // read database file using serde_json
    let db = Database::new(format!("db/{dirname}.json"));
    let mut all_paths = std::fs::read_dir(dir)
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
                let mut s = duration_to_char(duration).color(color);
                if bold {
                    s = s.bold();
                }
                if underline {
                    s = s.underline();
                }
                format!("{}", s)
            })
            .join("");
        let cnt = state.iter().filter(|x| x.2.solved()).count();
        let total = state.len();
        eprint!("{cnt:>3}/{total:>3} {summary}\r");
        log::info!("");
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
            let mut db = db.lock().unwrap_or_else(|_| {
                eprintln!("DB LOCK POISONED");
                exit(1)
            });

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

    let total_score = Mutex::new(Some(0));

    let process_path = |p: PathBuf| {
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
        db.lock()
            .unwrap_or_else(|_| {
                eprintln!("DB LOCK POISONED");
                exit(1)
            })
            .add_result(&p, duration, score);
        if score.is_some() {
            update(&state, &p, State::Solved(duration));
            if let Some(total_score) = total_score.lock().unwrap().as_mut() {
                *total_score += score.unwrap();
            }
        } else {
            update(&state, &p, State::Failed(duration));
            *total_score.lock().unwrap() = None;
        }
        print_state(&state.lock().unwrap());
    };

    if paths.len() == 1 {
        process_path(paths.pop().unwrap());
    } else {
        paths.into_iter().par_bridge().for_each_init(
            || {
                set_flags(&args.flags);
            },
            |_init, p| process_path(p),
        );
    }
    print_state(&state.lock().unwrap());
    eprintln!();

    if let Some(ts) = total_score.into_inner().unwrap() {
        eprintln!("TOTAL SCORE: {ts}");
    }
    Some(())
}
