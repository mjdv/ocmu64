run *args='':
    cargo run -r -- {{args}}
alias r := run
exact t='100':
    @cargo run -r --quiet -- --input input/exact --timelimit {{t}}
generate *args='':
    cargo run -r --bin generate -- {{args}}
flame *args='':
    cargo flamegraph --open --skip-after 'ocmu64::graph::Bb::branch_and_bound'  -- {{args}}
record *args='':
    cargo build -r --bin ocmu64
    perf record cargo run -r -- {{args}}
report:
    perf report
record_report *args='': (record args) report
alias p := record_report



# This requires `rustup target add x86_64-unknown-linux-musl`.
submit:
    cargo build --profile submit --target=x86_64-unknown-linux-musl
    ln -fns target/x86_64-unknown-linux-musl/submit/ocmu64 submit
