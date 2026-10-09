// SPDX-License-Identifier: MIT OR Apache-2.0
//! Reproducibility provenance written into sidecars / ODX extras.
//!
//! The default mode (`Minimal`) records compile-time facts (version, git SHA,
//! build timestamp) and run-time summaries (threads, duration) only. It omits
//! the command line, whose input paths often contain subject identifiers, the
//! host name and the start time; `Full` adds them.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::{BUILD_TIMESTAMP, GIT_SHA, VERSION};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
#[value(rename_all = "lower")]
pub enum ProvenanceMode {
    /// Version, build and run-time summary only. Default.
    Minimal,
    /// Additionally records the command line, host name and start time.
    Full,
    /// No provenance.
    None,
}

impl Default for ProvenanceMode {
    fn default() -> Self {
        Self::Minimal
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub tool: String,
    pub version: String,
    pub git_sha: String,
    pub build_timestamp: String,
    pub threads_used: usize,
    pub duration_secs: f64,

    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub argv: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub hostname: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub started_at_utc: Option<String>,
}

/// Captures start-time state. Call `finish` after the work loop with the
/// effective rayon thread count to produce the serializable `Provenance`.
pub struct ProvenanceBuilder {
    tool: &'static str,
    mode: ProvenanceMode,
    started: Instant,
    argv: Option<Vec<String>>,
    hostname: Option<String>,
    started_at_utc: Option<String>,
}

impl ProvenanceBuilder {
    /// Begin capture. Returns `None` for `ProvenanceMode::None`.
    pub fn new(tool: &'static str, mode: ProvenanceMode) -> Option<Self> {
        if matches!(mode, ProvenanceMode::None) {
            return None;
        }
        let (argv, hostname, started_at_utc) = if matches!(mode, ProvenanceMode::Full) {
            (
                Some(std::env::args().collect()),
                Some(detect_hostname()),
                Some(now_iso8601_utc()),
            )
        } else {
            (None, None, None)
        };
        Some(Self {
            tool,
            mode,
            started: Instant::now(),
            argv,
            hostname,
            started_at_utc,
        })
    }

    pub fn mode(&self) -> ProvenanceMode {
        self.mode
    }

    pub fn finish(self, threads_used: usize) -> Provenance {
        let duration_secs = self.started.elapsed().as_secs_f64();
        Provenance {
            tool: self.tool.to_string(),
            version: VERSION.to_string(),
            git_sha: GIT_SHA.to_string(),
            build_timestamp: BUILD_TIMESTAMP.to_string(),
            threads_used,
            duration_secs,
            argv: self.argv,
            hostname: self.hostname,
            started_at_utc: self.started_at_utc,
        }
    }
}

fn detect_hostname() -> String {
    if let Ok(h) = std::env::var("HOSTNAME") {
        if !h.is_empty() {
            return h;
        }
    }
    if let Ok(out) = std::process::Command::new("hostname").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
    }
    "unknown".to_string()
}

fn now_iso8601_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86_400;
    let (y, mo, d) = days_to_ymd(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn days_to_ymd(mut days: i64) -> (i64, u32, u32) {
    days += 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let mo = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if mo <= 2 { y + 1 } else { y };
    (y, mo, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_omits_phi_surface() {
        let b = ProvenanceBuilder::new("cs-fit", ProvenanceMode::Minimal).unwrap();
        let p = b.finish(4);
        assert!(p.argv.is_none());
        assert!(p.hostname.is_none());
        assert!(p.started_at_utc.is_none());
        assert_eq!(p.threads_used, 4);
        assert_eq!(p.tool, "cs-fit");
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("argv"));
        assert!(!json.contains("hostname"));
    }

    #[test]
    fn full_includes_runtime_fields() {
        let b = ProvenanceBuilder::new("cs-fit", ProvenanceMode::Full).unwrap();
        let p = b.finish(2);
        assert!(p.argv.is_some());
        assert!(p.hostname.is_some());
        assert!(p.started_at_utc.is_some());
    }

    #[test]
    fn none_returns_no_builder() {
        assert!(ProvenanceBuilder::new("cs-fit", ProvenanceMode::None).is_none());
    }
}
