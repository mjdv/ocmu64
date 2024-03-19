run *args='':
    cargo run -r -- {{args}}
exact t='100':
    @cargo run -r --quiet -- --input input/exact --timelimit {{t}}
generate *args='':
    cargo run -r --bin generate -- {{args}}
flamegraph *args='':
    cargo flamegraph --open --skip-after 'ocmu64::graph::Bb::branch_and_bound'  -- {{args}}
alias p := record
record *args='':
    cargo build -r
    perf record cargo run -r -- {{args}}
alias r := report
report:
    perf report
