// SPDX-License-Identifier: MIT OR Apache-2.0
//! MRtrix-format tissue response function loader.
//!
//! MRtrix's response files are plain text with one shell per row, ordered by
//! ascending b-value (the first row is the b=0 row by positional convention).
//! Each row holds the even-order zonal SH coefficients of the response on that
//! shell: `[r_0, r_2, r_4, …]`. Comments begin with `#`. All rows must share
//! the same column count, so a single `lmax` describes the whole file.
//!
//! Examples:
//!
//! ```text
//! # WM, lmax = 8 (b=0 row is isotropic, dwi row is anisotropic):
//! 1.234e-3 0 0 0 0
//! 5.67e-4 -1.2e-4 4.5e-5 -1.0e-5 2.0e-6
//! ```
//!
//! Isotropic tissues (GM, CSF) have a single column per row (lmax = 0).

use std::fs;
use std::path::Path;

use thiserror::Error;

use crate::Result;

/// Errors specific to response-file parsing / validation.
#[derive(Debug, Error)]
pub enum ResponseError {
    #[error("response file {path} is empty")]
    Empty { path: String },
    #[error("response file {path}: row {row} has {got} columns, expected {expected}")]
    RaggedRows { path: String, row: usize, got: usize, expected: usize },
    #[error("response file {path}: row {row} column {col} is not a number ({raw})")]
    BadNumber { path: String, row: usize, col: usize, raw: String },
    #[error(
        "response file {path}: expected isotropic tissue (lmax=0, 1 column per row), got {got} columns"
    )]
    NotIsotropic { path: String, got: usize },
    #[error(
        "response file {path}: anisotropic WM response on the b=0 row \
         (coefficient {coef} at index {idx} should be 0)"
    )]
    AnisotropicB0 { path: String, idx: usize, coef: f64 },
}

impl From<ResponseError> for crate::CsDmriError {
    fn from(e: ResponseError) -> Self {
        crate::CsDmriError::Other(e.to_string())
    }
}

/// A tissue response function: even-order zonal SH coefficients per shell.
#[derive(Debug, Clone)]
pub struct TissueResponse {
    /// `coeffs[shell_idx]` holds `[r_0, r_2, r_4, …]` for that shell. Shell 0
    /// is the b=0 row by MRtrix convention.
    pub coeffs: Vec<Vec<f64>>,
    /// Maximum even SH order represented in this file. Inferred from the
    /// (uniform) row width: `lmax = 2 · (ncols − 1)`.
    pub lmax: usize,
}

impl TissueResponse {
    /// Number of shells (rows) in the file.
    pub fn n_shells(&self) -> usize {
        self.coeffs.len()
    }

    /// Coefficient at order ℓ on shell `s`, or 0 if out of range.
    pub fn r(&self, shell: usize, l: usize) -> f64 {
        if l % 2 != 0 {
            return 0.0;
        }
        self.coeffs
            .get(shell)
            .and_then(|row| row.get(l / 2).copied())
            .unwrap_or(0.0)
    }

    /// Truncated copy with `lmax = min(self.lmax, max_lmax)`.
    pub fn clamp_lmax(&self, max_lmax: usize) -> TissueResponse {
        let new_lmax = self.lmax.min(max_lmax);
        let n_keep = new_lmax / 2 + 1;
        TissueResponse {
            coeffs: self
                .coeffs
                .iter()
                .map(|row| row.iter().take(n_keep).copied().collect())
                .collect(),
            lmax: new_lmax,
        }
    }

    /// Parse an MRtrix-format `.txt` response file.
    pub fn parse_mrtrix_txt(path: &Path) -> Result<Self> {
        let body = fs::read_to_string(path)?;
        Self::parse_mrtrix_text(&body, &path.display().to_string()).map_err(Into::into)
    }

    /// Parse from an in-memory string (split out for testing without filesystem).
    pub fn parse_mrtrix_text(body: &str, label: &str) -> std::result::Result<Self, ResponseError> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for (line_no, raw) in body.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut row = Vec::new();
            for (col_idx, tok) in line.split_whitespace().enumerate() {
                let v: f64 = tok.parse().map_err(|_| ResponseError::BadNumber {
                    path: label.to_string(),
                    row: line_no + 1,
                    col: col_idx + 1,
                    raw: tok.to_string(),
                })?;
                row.push(v);
            }
            rows.push(row);
        }
        if rows.is_empty() {
            return Err(ResponseError::Empty { path: label.to_string() });
        }
        let ncols = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != ncols {
                return Err(ResponseError::RaggedRows {
                    path: label.to_string(),
                    row: i + 1,
                    got: row.len(),
                    expected: ncols,
                });
            }
        }
        let lmax = 2 * (ncols - 1);
        Ok(Self { coeffs: rows, lmax })
    }

    /// Validate that this response is isotropic (lmax = 0). Used for GM/CSF.
    pub fn require_isotropic(&self, label: &str) -> std::result::Result<(), ResponseError> {
        if self.lmax != 0 {
            Err(ResponseError::NotIsotropic {
                path: label.to_string(),
                got: self.lmax / 2 + 1,
            })
        } else {
            Ok(())
        }
    }

    /// Validate that the b=0 row (shell 0) is isotropic — non-zero only at ℓ=0.
    /// Used for the WM response (its b=0 row should be a single isotropic
    /// coefficient, even though the file declares an anisotropic lmax).
    pub fn require_b0_isotropic(&self, label: &str) -> std::result::Result<(), ResponseError> {
        if let Some(row) = self.coeffs.first() {
            for (i, &c) in row.iter().enumerate().skip(1) {
                if c != 0.0 {
                    return Err(ResponseError::AnisotropicB0 {
                        path: label.to_string(),
                        idx: i,
                        coef: c,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wm_two_shell_lmax8() {
        let body = "\
            # WM SF response
            1.0e-3 0 0 0 0
            5.0e-4 -1.0e-4 3.0e-5 -8.0e-6 1.5e-6
        ";
        let r = TissueResponse::parse_mrtrix_text(body, "wm.txt").unwrap();
        assert_eq!(r.lmax, 8);
        assert_eq!(r.n_shells(), 2);
        assert_eq!(r.coeffs[0].len(), 5);
        assert_eq!(r.coeffs[1][0], 5.0e-4);
        assert_eq!(r.r(0, 0), 1.0e-3);
        assert_eq!(r.r(1, 4), 3.0e-5);
        assert_eq!(r.r(0, 1), 0.0); // odd ℓ always 0
        assert_eq!(r.r(0, 12), 0.0); // out-of-range ℓ
    }

    #[test]
    fn parses_isotropic_csf_one_column() {
        let body = "2.0e-3\n8.0e-4\n";
        let r = TissueResponse::parse_mrtrix_text(body, "csf.txt").unwrap();
        assert_eq!(r.lmax, 0);
        assert_eq!(r.n_shells(), 2);
        r.require_isotropic("csf.txt").unwrap();
    }

    #[test]
    fn rejects_ragged_rows() {
        let body = "1.0 0 0 0 0\n5.0e-4 -1.0e-4 3.0e-5\n";
        let err = TissueResponse::parse_mrtrix_text(body, "bad.txt").unwrap_err();
        assert!(matches!(err, ResponseError::RaggedRows { row: 2, got: 3, expected: 5, .. }));
    }

    #[test]
    fn rejects_empty_file() {
        let err = TissueResponse::parse_mrtrix_text("# only a comment\n\n", "x.txt").unwrap_err();
        assert!(matches!(err, ResponseError::Empty { .. }));
    }

    #[test]
    fn rejects_bad_number() {
        let err =
            TissueResponse::parse_mrtrix_text("1.0 0 not_a_number 0 0\n", "x.txt").unwrap_err();
        assert!(matches!(err, ResponseError::BadNumber { row: 1, col: 3, .. }));
    }

    #[test]
    fn require_isotropic_rejects_anisotropic_file() {
        let r = TissueResponse::parse_mrtrix_text("1.0 0 0 0 0\n0.5 0 0 0 0\n", "x.txt").unwrap();
        assert!(r.require_isotropic("x.txt").is_err());
    }

    #[test]
    fn require_b0_isotropic_rejects_nonzero_l_gt_0() {
        let r = TissueResponse::parse_mrtrix_text(
            "1.0e-3 1.0e-4 0 0 0\n5.0e-4 -1.0e-4 3.0e-5 -8.0e-6 1.5e-6\n",
            "wm.txt",
        )
        .unwrap();
        let err = r.require_b0_isotropic("wm.txt").unwrap_err();
        assert!(matches!(err, ResponseError::AnisotropicB0 { idx: 1, .. }));
    }

    #[test]
    fn clamp_lmax_truncates_columns() {
        let r = TissueResponse::parse_mrtrix_text(
            "1.0 0 0 0 0\n5.0e-4 -1.0e-4 3.0e-5 -8.0e-6 1.5e-6\n",
            "wm.txt",
        )
        .unwrap();
        let r6 = r.clamp_lmax(6);
        assert_eq!(r6.lmax, 6);
        assert_eq!(r6.coeffs[1].len(), 4);
        assert_eq!(r6.coeffs[1][3], -8.0e-6);
        let r10 = r.clamp_lmax(10); // already smaller, no expansion
        assert_eq!(r10.lmax, 8);
    }
}
