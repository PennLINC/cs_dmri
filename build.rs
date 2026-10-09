// SPDX-License-Identifier: MIT OR Apache-2.0
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let git_sha = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    let git_sha_full = if dirty && git_sha != "unknown" {
        format!("{git_sha}-dirty")
    } else {
        git_sha
    };

    let build_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| iso8601_utc(d.as_secs()))
        .unwrap_or_else(|_| "unknown".to_string());

    println!("cargo:rustc-env=CS_DMRI_GIT_SHA={git_sha_full}");
    println!("cargo:rustc-env=CS_DMRI_BUILD_TS={build_ts}");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");
    println!("cargo:rerun-if-changed=build.rs");
}

fn iso8601_utc(epoch_secs: u64) -> String {
    let secs = epoch_secs % 60;
    let mins = (epoch_secs / 60) % 60;
    let hours = (epoch_secs / 3600) % 24;
    let days_since_epoch = epoch_secs / 86_400;
    let (y, m, d) = days_to_ymd(days_since_epoch as i64);
    format!("{y:04}-{m:02}-{d:02}T{hours:02}:{mins:02}:{secs:02}Z")
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
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
