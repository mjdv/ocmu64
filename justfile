run *args='':
    cargo run -r -- {{args}}
generate *args='':
    cargo run -r --bin generate -- {{args}}
flamegraph *args='':
    cargo flamegraph -- {{args}}
alias p := record
record *args='':
    cargo build -r
    perf record cargo run -r -- {{args}}
alias r := report
report:
    perf report
