use std::path::PathBuf;

use clap::Parser;
use ocmu64::generate::GraphType;

#[derive(Parser)]
#[clap(disable_help_flag = true)]
struct Args {
    /// The type of graph to generate.
    #[clap(subcommand)]
    graph_type: GraphType,
    /// Optional path to output file, or stdout by default.
    #[clap(short, long)]
    output: Option<PathBuf>,
    #[clap(short, long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let g = args.graph_type.generate(args.seed);
    match args.output {
        None => g.to_stdout(),
        Some(f) => g.to_file(&f),
    }
    .unwrap();
}
