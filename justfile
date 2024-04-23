run *args='':
    cargo run -r -- {{args}}
exact t='100':
    @cargo run -r --quiet -- --input input/exact --timelimit {{t}}
generate *args='':
    cargo run -r --bin generate -- {{args}}
flame *args='':
    cargo flamegraph --open --skip-after 'ocmu64::graph::Bb::branch_and_bound'  -- {{args}}
alias p := record
record *args='':
    cargo build -r
    perf record cargo run -r -- {{args}}
alias r := report
report:
    perf report


# This requires `rustup target add x86_64-unknown-linux-musl`.
submit:
    cargo build --profile submit --target=x86_64-unknown-linux-musl
    ln -s target/x86_64-unknown-linux-musl/submit/ocmu64 submit
