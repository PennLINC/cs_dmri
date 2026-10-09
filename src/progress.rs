// SPDX-License-Identifier: MIT OR Apache-2.0
//! Time-based progress heartbeat for long parallel loops.
//!
//! Single producer model: each rayon worker calls `tick()` after finishing a
//! voxel; a sidecar reporter thread wakes periodically, reads the atomic
//! counter, and prints a one-line status to stderr. Plain lines, no carriage
//! returns — safe for non-TTY job logs (SLURM, sbatch, etc.).
//!
//! Quiet mode short-circuits everything: `tick()` becomes a relaxed atomic
//! increment with no reporter thread, and `finish()` prints nothing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub struct Heartbeat {
    inner: Arc<Inner>,
    reporter: Option<JoinHandle<()>>,
}

struct Inner {
    label: &'static str,
    total: usize,
    counter: AtomicUsize,
    stop: AtomicBool,
    started: Instant,
    quiet: bool,
}

impl Heartbeat {
    /// Begin a heartbeat for a parallel loop covering `total` units. If
    /// `quiet`, no reporter thread is spawned and `finish()` prints nothing.
    /// `interval` must be positive; pass at least 1 second for sane logs.
    pub fn new(label: &'static str, total: usize, interval: Duration, quiet: bool) -> Self {
        let inner = Arc::new(Inner {
            label,
            total,
            counter: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            started: Instant::now(),
            quiet,
        });
        let reporter = if quiet || total == 0 {
            None
        } else {
            let inner_cloned = Arc::clone(&inner);
            Some(thread::spawn(move || run_reporter(inner_cloned, interval)))
        };
        Self { inner, reporter }
    }

    /// Increment the completed-unit counter. Cheap (relaxed atomic add).
    /// Safe to call from any thread.
    #[inline]
    pub fn tick(&self) {
        self.inner.counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment by a batch of completed units.
    #[inline]
    pub fn tick_n(&self, n: usize) {
        self.inner.counter.fetch_add(n, Ordering::Relaxed);
    }

    /// Stop the reporter thread and print a final summary line.
    pub fn finish(mut self) {
        self.inner.stop.store(true, Ordering::Release);
        if let Some(h) = self.reporter.take() {
            let _ = h.join();
        }
        if !self.inner.quiet {
            let done = self.inner.counter.load(Ordering::Relaxed);
            let elapsed = self.inner.started.elapsed();
            eprintln!(
                "[{label}] done {done}/{total} ({pct:.1}%) elapsed={elapsed}",
                label = self.inner.label,
                done = done,
                total = self.inner.total,
                pct = pct(done, self.inner.total),
                elapsed = format_hms(elapsed),
            );
        }
    }
}

impl Drop for Heartbeat {
    fn drop(&mut self) {
        // Defensive: if the caller forgot finish(), still stop the thread.
        if self.reporter.is_some() {
            self.inner.stop.store(true, Ordering::Release);
            if let Some(h) = self.reporter.take() {
                let _ = h.join();
            }
        }
    }
}

fn run_reporter(inner: Arc<Inner>, interval: Duration) {
    let tick = Duration::from_millis(200).min(interval);
    let mut next = Instant::now() + interval;
    loop {
        if inner.stop.load(Ordering::Acquire) {
            return;
        }
        thread::sleep(tick);
        if Instant::now() >= next {
            let done = inner.counter.load(Ordering::Relaxed);
            let elapsed = inner.started.elapsed();
            let eta = eta_secs(done, inner.total, elapsed);
            eprintln!(
                "[{label}] {done}/{total} ({pct:.1}%) elapsed={elapsed} eta={eta}",
                label = inner.label,
                done = done,
                total = inner.total,
                pct = pct(done, inner.total),
                elapsed = format_hms(elapsed),
                eta = eta,
            );
            next = Instant::now() + interval;
        }
    }
}

fn pct(done: usize, total: usize) -> f64 {
    if total == 0 {
        100.0
    } else {
        (done as f64) * 100.0 / (total as f64)
    }
}

fn eta_secs(done: usize, total: usize, elapsed: Duration) -> String {
    if done == 0 || done >= total {
        return "--:--:--".to_string();
    }
    let rate = elapsed.as_secs_f64() / done as f64;
    let remaining = (total - done) as f64 * rate;
    format_hms(Duration::from_secs_f64(remaining))
}

fn format_hms(d: Duration) -> String {
    let total = d.as_secs();
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    format!("{h:02}:{m:02}:{s:02}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quiet_mode_does_not_panic_or_print() {
        let hb = Heartbeat::new("t", 100, Duration::from_secs(60), true);
        for _ in 0..100 {
            hb.tick();
        }
        hb.finish();
    }

    #[test]
    fn tick_counts_match() {
        let hb = Heartbeat::new("t", 10, Duration::from_secs(60), true);
        hb.tick_n(7);
        hb.tick();
        assert_eq!(hb.inner.counter.load(Ordering::Relaxed), 8);
        hb.finish();
    }

    #[test]
    fn reporter_stops_promptly_on_finish() {
        let hb = Heartbeat::new("t", 1000, Duration::from_secs(30), false);
        hb.tick();
        let t0 = Instant::now();
        hb.finish();
        // Should not block on the next 30s tick.
        assert!(t0.elapsed() < Duration::from_secs(2));
    }
}
