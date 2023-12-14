run *args='':
    cargo run -r -- {{args}}
generate *args='':
    cargo run -r --bin generate -- {{args}}
