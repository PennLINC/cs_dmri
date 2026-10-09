// SPDX-License-Identifier: MIT OR Apache-2.0
//! `cs-qc`: image-quality metrics for a DWI series, as JSON and/or a one-row
//! TSV (with a BIDS-style JSON data dictionary for the TSV's columns).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;

use cs_dmri::dti::{DtiFitConfig, RestoreConfig, fit_volume_restore_reporting};
use cs_dmri::io::atomic_write;
use cs_dmri::io::dwi::load_dwi;
use cs_dmri::qc::{
    CoherenceOptions, OutlierSliceOptions, QcOptions, assess, contrast_grade, fixel_coherence,
    qc_columns, voxel_size,
};
use cs_dmri::qspace::{BvecFrame, TORTOISE_DEFAULT_GMAX};
use cs_dmri::{Heartbeat, configure_rayon_threads};

#[derive(Parser, Debug)]
#[command(version, about = "DWI image-quality metrics (NDC, DWI contrast ratio, outlier slices, fixel coherence)")]
struct Cli {
    /// 4D DWI NIfTI input.
    #[arg(long)]
    dwi: PathBuf,
    /// FSL bval file.
    #[arg(long)]
    bval: PathBuf,
    /// FSL bvec file.
    #[arg(long)]
    bvec: PathBuf,
    /// Brain mask NIfTI on the DWI grid. Without one the masked metrics use a
    /// fallback mask (mean b=0 above 1% of its maximum), recorded as
    /// `mask_source: auto-b0` in the JSON.
    #[arg(long)]
    mask: Option<PathBuf>,
    /// Volumes with b at or below this are b=0.
    #[arg(long, default_value_t = 50.0)]
    b0_threshold: f64,
    /// Spatial axis (0, 1 or 2) that outlier slices are taken along.
    #[arg(long, default_value_t = 2)]
    slice_axis: usize,
    /// Skip the tensor fit and fixel coherence.
    #[arg(long)]
    no_coherence: bool,
    /// Write the full report (metrics, flagged slices, settings) as JSON.
    #[arg(long)]
    output_json: Option<PathBuf>,
    /// Write the flat metrics as a one-row TSV, plus `<stem>.json` describing
    /// each column.
    #[arg(long)]
    output_tsv: Option<PathBuf>,
    /// Prefix for TSV column names (e.g. `raw_`).
    #[arg(long, default_value = "")]
    prefix: String,
    /// Cap rayon's worker threads.
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    overwrite: bool,
    #[arg(long)]
    quiet: bool,
}

fn main() -> Result<()> {
    let args = Cli::parse();
    if args.output_json.is_none() && args.output_tsv.is_none() && args.quiet {
        anyhow::bail!("nothing to do: pass --output-json and/or --output-tsv, or drop --quiet");
    }
    configure_rayon_threads(args.threads)?;
    let dwi = load_dwi(
        &args.dwi,
        &args.bval,
        &args.bvec,
        args.mask.as_deref(),
        None,
        None,
        Some(TORTOISE_DEFAULT_GMAX),
        BvecFrame::WorldRas,
    )
    .with_context(|| "failed to load DWI bundle")?;
    let affine = odx_rs::read_reference_affine(&args.dwi)
        .map_err(|e| anyhow::anyhow!("read affine from {:?}: {e}", args.dwi))?;
    let mask_source = if args.mask.is_some() { "provided" } else { "auto-b0" };

    let opts = QcOptions {
        b0_threshold: args.b0_threshold,
        outlier_slices: OutlierSliceOptions {
            axis: args.slice_axis,
            ..OutlierSliceOptions::default()
        },
    };
    let report = assess(dwi.data.view(), &dwi.gtab, Some(dwi.mask.view()), &opts)?;

    let coherence = if args.no_coherence {
        None
    } else {
        let n = dwi.mask.iter().filter(|&&m| m).count();
        let hb = Heartbeat::new("cs-qc dti", n, Duration::from_secs(30), args.quiet);
        let dti = fit_volume_restore_reporting(
            &dwi,
            &RestoreConfig::default(),
            DtiFitConfig { compute_diagnostics: false },
            || hb.tick(),
        )?;
        hb.finish();
        Some(fixel_coherence(
            dti.principal_dir.view(),
            dti.fa.view(),
            dwi.mask.view(),
            affine,
            true,
            &CoherenceOptions::default(),
        )?)
    };

    let row = report.row(Some(voxel_size(&affine)), coherence.as_ref().and_then(|c| c.coherence));
    let warnings = report.warnings();
    if !args.quiet {
        let f = |v: Option<f64>| v.map_or("n/a".into(), |v| format!("{v:.4}"));
        eprintln!(
            "[cs-qc] NDC {} (masked {}), DWI contrast ratio {} (masked {}{}), outlier slices {}, fixel coherence {}",
            f(report.ndc),
            f(report.ndc_masked),
            f(report.dwi_contrast_ratio),
            f(report.dwi_contrast_ratio_masked),
            report.dwi_contrast_ratio_masked.map_or(String::new(), |v| format!(", {}", contrast_grade(v))),
            report.n_outlier_slices(),
            f(coherence.as_ref().and_then(|c| c.coherence)),
        );
    }
    for w in &warnings {
        eprintln!("[cs-qc] WARNING: {w}");
    }

    if let Some(path) = &args.output_json {
        let columns: serde_json::Map<String, serde_json::Value> =
            row.iter().map(|(k, v)| (k.to_string(), serde_json::json!(v))).collect();
        let doc = serde_json::json!({
            "metrics": columns,
            "report": report,
            "fixel_coherence": coherence,
            "mask_source": mask_source,
            "warnings": warnings,
            "cs_dmri_version": cs_dmri::VERSION,
        });
        write_text(path, &serde_json::to_string_pretty(&doc)?, args.overwrite)?;
    }
    if let Some(path) = &args.output_tsv {
        let header: Vec<String> = row.iter().map(|(k, _)| format!("{}{k}", args.prefix)).collect();
        let values: Vec<String> = row
            .iter()
            .map(|(_, v)| v.filter(|x| x.is_finite()).map_or("n/a".into(), |x| format!("{x}")))
            .collect();
        write_text(path, &format!("{}\n{}\n", header.join("\t"), values.join("\t")), args.overwrite)?;
        let dict: serde_json::Map<String, serde_json::Value> = qc_columns()
            .into_iter()
            .map(|c| (format!("{}{}", args.prefix, c.name), serde_json::to_value(&c).unwrap()))
            .map(|(k, mut v)| {
                v.as_object_mut().unwrap().remove("name");
                (k, v)
            })
            .collect();
        let dict_path = path.with_extension("").with_extension("json");
        write_text(&dict_path, &serde_json::to_string_pretty(&dict)?, args.overwrite)?;
    }
    Ok(())
}

fn write_text(path: &std::path::Path, text: &str, overwrite: bool) -> Result<()> {
    atomic_write(path, overwrite, |tmp| std::fs::write(tmp, text).map_err(Into::into))
        .with_context(|| format!("write {}", path.display()))
}
