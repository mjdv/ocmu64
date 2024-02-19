use colored::Colorize;
use log::error;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::get_flag;

/// Data stored per testcase.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Data {
    results: HashMap<PathBuf, InputResults>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InputResults {
    /// Path to the input, eg input/exact/01.gr
    path: PathBuf,
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
    duration: u64,
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
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let data = if path.as_ref().exists() {
            let file = std::fs::File::open(&path).unwrap();
            serde_json::from_reader(std::io::BufReader::new(file)).unwrap()
        } else {
            Data {
                results: HashMap::new(),
            }
        };

        Self {
            path: path.as_ref().to_path_buf(),
            data,
        }
    }

    pub fn save(&self) {
        let prefix = self.path.parent().unwrap();
        std::fs::create_dir_all(prefix).unwrap();
        let file = std::fs::File::create(&self.path).unwrap();
        serde_json::to_writer(std::io::BufWriter::new(file), &self.data).unwrap();
    }

    pub fn get_score(&self, input: &Path) -> Option<u64> {
        self.data.results.get(input).and_then(|x| x.score)
    }

    /// When unsolved: max duration over all runs.
    /// When solved: fastest run giving the optimal score.
    pub fn get_duration(&self, input: &Path) -> u64 {
        let results = self.data.results.get(input);
        let Some(results) = results else {
            return 0;
        };
        if results.score.is_some() {
            results
                .runs
                .iter()
                .filter(|x| x.score == results.score)
                .map(|x| x.duration)
                .min()
                .unwrap()
        } else {
            results.runs.iter().map(|x| x.duration).max().unwrap()
        }
    }

    pub fn add_result(&mut self, input: &Path, duration: u64, score: Option<u64>) {
        let results = self
            .data
            .results
            .entry(input.to_path_buf())
            .or_insert_with(|| InputResults {
                path: input.to_path_buf(),
                score: None,
                runs: Vec::new(),
            });

        let result = RunResult {
            start: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false),
            duration,
            score,
        };

        results.runs.push(result);
        if let Some(new_score) = score {
            if let Some(existing_score) = &results.score {
                if get_flag("update_db_score") {
                    if new_score != *existing_score {
                        let msg = format!(
                            "\nUpdating score for {} from {} to {}.\n",
                            input.display(),
                            existing_score,
                            new_score
                        );
                        error!("{}", msg.red());
                        results.score = Some(new_score);
                    }
                } else {
                    assert_eq!(
                        new_score,
                        *existing_score,
                        "New score {new_score} does not equal existing score {existing_score} for {}.",
                        results.path.display(),
                    );
                    results.score = Some(new_score.min(*existing_score));
                }
            } else {
                results.score = Some(new_score);
            }
        }
        self.save();
    }
}
