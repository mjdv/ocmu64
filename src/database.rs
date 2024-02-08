use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::PathBuf,
    time::{Duration, SystemTime},
};

/// Data stored per testcase.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Data {
    results: HashMap<String, InputResults>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InputResults {
    /// Path to the input, eg input/exact/01.gr
    path: String,
    /// The best score.
    score: Option<u64>,
    /// All runs.
    runs: Vec<RunResult>,
}

/// The result of a single run.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunResult {
    /// Time the run finished.
    start: String,
    /// Duration of the run, in seconds.
    duration: f32,
    /// The score of the solution, if the run finished.
    score: Option<u64>,
}

pub struct Database {
    /// Path to the json file.
    path: PathBuf,
    /// The data.
    data: Data,
}

impl Database {
    pub fn new(path: PathBuf) -> Self {
        let data = if path.exists() {
            let file = std::fs::File::open(&path).unwrap();
            serde_json::from_reader(std::io::BufReader::new(file)).unwrap()
        } else {
            Data {
                results: HashMap::new(),
            }
        };

        Self { path, data }
    }

    pub fn save(&self) {
        let file = std::fs::File::create(&self.path).unwrap();
        serde_json::to_writer(std::io::BufWriter::new(file), &self.data).unwrap();
    }

    pub fn get_score(&self, input: &str) -> Option<u64> {
        self.data.results.get(input).and_then(|x| x.score)
    }

    pub fn get_max_duration(&self, input: &str) -> f32 {
        self.data
            .results
            .get(input)
            .map(|x| x.runs.iter().map(|x| x.duration).fold(0.0, f32::max))
            .unwrap_or(0.0)
    }

    pub fn add_result(&mut self, input: String, duration: Duration, score: Option<u64>) {
        let results = self
            .data
            .results
            .entry(input.clone())
            .or_insert_with(|| InputResults {
                path: input,
                score: None,
                runs: Vec::new(),
            });

        let result = RunResult {
            start: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
            duration: duration.as_secs_f32(),
            score,
        };

        results.runs.push(result);
        if let Some(new_score) = score {
            if let Some(existing_score) = &results.score {
                results.score = Some(new_score.min(*existing_score));
            } else {
                results.score = Some(new_score);
            }
        }
        self.save();
    }
}
