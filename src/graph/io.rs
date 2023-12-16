use super::*;
use std::{
    fs::File,
    io::{stdin, stdout, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

impl Graph {
    fn to_stream<W: Write>(&self, writer: W) -> Result<(), std::io::Error> {
        let mut writer = BufWriter::new(writer);
        let edges = self.connections_a.iter().map(|x| x.len()).sum::<usize>();
        writeln!(writer, "p ocr {} {} {}", self.a.0, self.b.0, edges)?;
        for i in NodeA(0)..self.a {
            for j in &self.connections_a[i] {
                writeln!(writer, "{} {}", i.0 + 1, j.0 + self.a.0 + 1)?;
            }
        }
        Ok(())
    }

    pub fn to_file(&self, file_path: &Path) -> Result<(), std::io::Error> {
        let file = File::create(file_path)?;
        self.to_stream(file)
    }

    pub fn to_stdout(&self) -> Result<(), std::io::Error> {
        self.to_stream(stdout().lock())
    }

    fn from_stream<T: BufRead>(stream: T) -> Result<Self, std::io::Error> {
        let mut a = NodeA::default();
        let mut connections_a: VecA<Vec<NodeB>> = VecA::default();
        let mut connections_b: VecB<Vec<NodeA>> = VecB::default();

        for line in stream.lines() {
            let line = line?;
            if line.starts_with('c') {
                continue;
            } else if line.starts_with('p') {
                let words = line.split(' ').collect::<Vec<&str>>();
                a = NodeA(words[2].parse().unwrap());
                let b = NodeB(words[3].parse().unwrap());
                connections_a = VecA::new(a);
                connections_b = VecB::new(b);
            } else {
                let mut words = line.split_ascii_whitespace();
                let x = NodeA(words.next().unwrap().parse::<usize>().unwrap() - 1);
                let y = NodeB(words.next().unwrap().parse::<usize>().unwrap() - a.0 - 1);
                connections_a[x].push(y);
                connections_b[y].push(x);
            }
        }

        let graph = GraphBuilder::new(connections_a, connections_b).build();
        Ok(graph)
    }

    // Reads a graph from a file (in PACE format).
    pub fn from_file(file_path: &Path) -> Result<Self, std::io::Error> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        Self::from_stream(reader)
    }
    // Reads a graph from stdin (in PACE format).
    pub fn from_stdin() -> Result<Self, std::io::Error> {
        Self::from_stream(stdin().lock())
    }
}