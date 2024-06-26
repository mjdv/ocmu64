build:
    cargo build -r --bin ocmu64
alias b := build
run *args='':
    cargo run -r -- {{args}}
alias r := run
exact t='100':
    @cargo run -r --quiet -F exact -- --input input/exact --timelimit {{t}}
generate *args='':
    cargo run -r --bin generate -- {{args}}
flame *args='':
    cargo flamegraph --open --skip-after 'ocmu64::graph::Bb::branch_and_bound'  -- {{args}}
record *args='': build
    perf record cargo run -r -- {{args}}
report:
    perf report
record_report *args='': (record args) report
alias p := record_report



# This requires `rustup target add x86_64-unknown-linux-musl`.
submit-parameterized:
    cargo build --profile submit --target=x86_64-unknown-linux-musl
    ln -fns target/x86_64-unknown-linux-musl/submit/ocmu64 submit

submit-exact:
    cargo build --profile submit --target=x86_64-unknown-linux-musl -F exact
    ln -fns target/x86_64-unknown-linux-musl/submit/ocmu64 submit
