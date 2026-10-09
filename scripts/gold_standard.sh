#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Gold-standard MRtrix MSMT-CSD pipeline + ODX cross-comparison harness.
#
# Mirrors qsirecon's `mrtrix_multishell_msmt_noACT` pipeline (mrconvert,
# dwi2response dhollander, dwi2fod msmt_csd, mtnormalise) and appends
# fod2fixel + `odx convert` to land the WM FOD and fixels in a single ODX
# archive. Then runs `odx compare` per HASC config to produce:
#
#   <out>/<pair>/<config>/cs_direct_vs_gold/    coeffs.odx     vs abcd_gold.odx
#   <out>/<pair>/<config>/cs_roundtrip_vs_gold/ predicted_gold vs abcd_gold.odx
#
# Every DWI must ship with an MRtrix-native gradient table (`<stem>.b`,
# four-column `gx gy gz b` per row). The script never falls back to
# `-fslgrad` — FSL bvec orientation can silently rotate the gradient frame
# and corrupt the FOD.

set -euo pipefail

ODX_BIN="${ODX_BIN:-odx}"
DEFAULT_BENCH_ROOT="${HOME}/cs-bench-csdsi/focused"
DEFAULT_CONFIGS="l2_baseline,l1_ratio_1e-3,l1_pathbic_n20,l1_pathbic_ro4,l1_pathbic_nn"
DEFAULT_THREADS=10

usage() {
    cat <<EOF
Usage: $0 --bundles-root DIR [options]

Required:
  --bundles-root DIR     BIDS-like dir with *desc-preproc_dwi.nii(.gz),
                         matching *desc-preproc_dwi.b (MRtrix grad), and
                         *desc-brain_mask*.nii(.gz).

Optional:
  --bench-root DIR       cs-bench output root (default: $DEFAULT_BENCH_ROOT).
  --out-root DIR         Where to write gold outputs + comparisons
                         (default: <bench-root>/gold).
  --configs CSV          Comma-separated cs-odf configs to process
                         (default: $DEFAULT_CONFIGS).
  --filter-bundles CSV   Only process pairs where the HASC bundle name
                         contains any of these substrings.
  --threads N            MRtrix threads (default: $DEFAULT_THREADS).
  --force                Rebuild even if .done sentinels are present.
  -h, --help             Show this help.
EOF
}

BUNDLES_ROOT=""
BENCH_ROOT=""
OUT_ROOT=""
CONFIGS_CSV="$DEFAULT_CONFIGS"
FILTER_CSV=""
THREADS="$DEFAULT_THREADS"
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundles-root)   BUNDLES_ROOT="$2"; shift 2;;
        --bench-root)     BENCH_ROOT="$2"; shift 2;;
        --out-root)       OUT_ROOT="$2"; shift 2;;
        --configs)        CONFIGS_CSV="$2"; shift 2;;
        --filter-bundles) FILTER_CSV="$2"; shift 2;;
        --threads)        THREADS="$2"; shift 2;;
        --force)          FORCE=1; shift;;
        -h|--help)        usage; exit 0;;
        *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
    esac
done

if [[ -z "$BUNDLES_ROOT" ]]; then
    echo "ERROR: --bundles-root is required" >&2; usage >&2; exit 2
fi
BENCH_ROOT="${BENCH_ROOT:-$DEFAULT_BENCH_ROOT}"
OUT_ROOT="${OUT_ROOT:-$BENCH_ROOT/gold}"
mkdir -p "$OUT_ROOT"

FILTER_RE=""
if [[ -n "$FILTER_CSV" ]]; then
    FILTER_RE="$(printf '%s' "$FILTER_CSV" | tr ',' '|')"
fi

for tool in mrconvert dwi2response dwi2fod mtnormalise fod2fixel; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: required MRtrix3 tool not on PATH: $tool" >&2
        exit 3
    fi
done
if [[ ! -x "$ODX_BIN" ]]; then
    echo "ERROR: odx binary not found or not executable: $ODX_BIN" >&2
    echo "       set ODX_BIN env var or build with: cargo build --release --bin odx" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Bundle discovery
#
# Emits TSV rows: acq \t name \t dwi \t b \t mask \t pair_key
#   acq      "HASC" or "ABCD" (other acqs ignored)
#   name     BIDS-ish bundle name (sub-X[_ses-Y][_acq-Z][_run-N])
#   pair_key name with `_acq-*` and `_run-*` stripped — used to match HASC↔ABCD
# ---------------------------------------------------------------------------
discover_bundles_tsv() {
    local root="$1"
    {
        find "$root" -name "*desc-preproc_dwi.nii.gz" -type f 2>/dev/null
        find "$root" -name "*desc-preproc_dwi.nii"    -type f 2>/dev/null
    } | sort -u | while IFS= read -r dwi; do
        local stem="${dwi%.nii.gz}"; stem="${stem%.nii}"
        local b="${stem}.b"
        if [[ ! -f "$b" ]]; then
            echo "WARN: skip $dwi: missing .b sibling at $b" >&2
            continue
        fi
        local mask=""
        local mask_stem="${stem//desc-preproc_dwi/desc-brain_mask}"
        for ext in .nii.gz .nii; do
            if [[ -f "${mask_stem}${ext}" ]]; then mask="${mask_stem}${ext}"; break; fi
        done
        if [[ -z "$mask" ]]; then
            mask="$(ls "$(dirname "$dwi")"/*desc-brain*mask*.nii* 2>/dev/null | head -n1)"
        fi
        if [[ -z "$mask" ]]; then
            echo "WARN: skip $dwi: no brain mask sibling" >&2
            continue
        fi
        local fname; fname="$(basename "$dwi")"
        local name=""
        for key in sub- ses- acq- run-; do
            local m
            m="$(printf '%s' "$fname" | grep -oE "${key}[A-Za-z0-9]+" | head -n1 || true)"
            [[ -n "$m" ]] && name+="${m}_"
        done
        name="${name%_}"
        [[ -z "$name" ]] && name="$(basename "$stem")"
        local acq_tag
        acq_tag="$(printf '%s' "$name" | grep -oE 'acq-[A-Za-z0-9]+' | sed 's/acq-//' || true)"
        local acq=""
        case "$acq_tag" in
            ABCD) acq=ABCD;;
            HASC*) acq=HASC;;
            *) continue;;
        esac
        local pair_key
        pair_key="$(printf '%s' "$name" | sed -E 's/_acq-[A-Za-z0-9]+//; s/_run-[A-Za-z0-9]+//')"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$acq" "$name" "$dwi" "$b" "$mask" "$pair_key"
    done
}

# ---------------------------------------------------------------------------
# MRtrix MSMT recon + odx convert. Idempotent via .done sentinel.
# Args: dwi b mask out_dir tag
# ---------------------------------------------------------------------------
run_mrtrix_recon() {
    local dwi="$1" b="$2" mask="$3" out="$4" tag="$5"
    local sentinel="$out/.done"
    if [[ -f "$sentinel" && $FORCE -eq 0 ]]; then
        echo "[$tag] cached"; return 0
    fi
    mkdir -p "$out"
    local nthr=(-nthreads "$THREADS")
    echo "[$tag] mrconvert"
    mrconvert "${nthr[@]}" -force -quiet -grad "$b" "$dwi" "$out/dwi.mif"
    echo "[$tag] dwi2response dhollander"
    rm -rf "$out/_scratch_dwi2response"
    # No -lmax: dhollander expects one entry per b-value shell, and the data
    # may have 4 (b=0,1000,2000,3000) or 5 (with b=500). Letting MRtrix pick
    # per-shell maxima is more robust than hard-coding a count.
    dwi2response dhollander "${nthr[@]}" -force -quiet \
        -mask "$mask" \
        -voxels "$out/response_voxels.mif" \
        -scratch "$out/_scratch_dwi2response" \
        "$out/dwi.mif" "$out/wm.txt" "$out/gm.txt" "$out/csf.txt"
    rm -rf "$out/_scratch_dwi2response"
    echo "[$tag] dwi2fod msmt_csd"
    dwi2fod msmt_csd "${nthr[@]}" -force -quiet \
        -lmax 8,8,8 -mask "$mask" \
        "$out/dwi.mif" \
        "$out/wm.txt"  "$out/wmfod.mif" \
        "$out/gm.txt"  "$out/gmfod.mif" \
        "$out/csf.txt" "$out/csffod.mif"
    echo "[$tag] mtnormalise"
    mtnormalise "${nthr[@]}" -force -quiet -mask "$mask" \
        "$out/wmfod.mif"  "$out/wmfod_norm.mif" \
        "$out/gmfod.mif"  "$out/gmfod_norm.mif" \
        "$out/csffod.mif" "$out/csffod_norm.mif"
    # Both cs-odf and `odx convert` now canonicalize NIfTI inputs to RAS+ on
    # ingest, so input strides are no longer load-bearing for voxel
    # alignment. Convert directly to NIfTI without inheriting DWI strides.
    echo "[$tag] mrconvert wmfod -> nii"
    mrconvert "${nthr[@]}" -force -quiet \
        "$out/wmfod_norm.mif" "$out/wmfod_norm.nii.gz"
    echo "[$tag] fod2fixel"
    rm -rf "$out/fixels"
    fod2fixel "${nthr[@]}" -force -quiet -mask "$mask" \
        -afd afd.mif -peak_amp peak_amp.mif -disp dispersion.mif \
        "$out/wmfod_norm.nii.gz" "$out/fixels"
    echo "[$tag] mrconvert index -> nii"
    if [[ -f "$out/fixels/index.mif" ]]; then
        mrconvert "${nthr[@]}" -force -quiet \
            "$out/fixels/index.mif" "$out/fixels/index.nii.gz"
        rm -f "$out/fixels/index.mif"
    fi
    echo "[$tag] odx convert (archive)"
    rm -f "$out/gold.odx"
    "$ODX_BIN" convert "$out/wmfod_norm.nii.gz" "$out/gold.odx" \
        --fixel-dir "$out/fixels" --odx-layout archive \
        --overwrite --quiet
    touch "$sentinel"
    echo "[$tag] done"
}

# Args: a b out_dir tag
run_compare() {
    local a="$1" b="$2" out="$3" tag="$4"
    local sentinel="$out/.done"
    if [[ -f "$sentinel" && $FORCE -eq 0 ]]; then
        echo "[$tag] cached"; return 0
    fi
    if [[ ! -e "$a" ]]; then echo "[$tag] SKIP: A missing: $a" >&2; return 0; fi
    if [[ ! -e "$b" ]]; then echo "[$tag] SKIP: B missing: $b" >&2; return 0; fi
    mkdir -p "$out"
    # No --primary-dpf: both cs-odf and the MRtrix loader emit `amplitude` as
    # the canonical peak-amplitude DPF, which is the first entry in the
    # comparator's auto-detect chain (amplitude → afd → qa).
    "$ODX_BIN" compare --a "$a" --b "$b" --out-dir "$out"
    touch "$sentinel"
    echo "[$tag] done"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_TSV="$(mktemp -t gold_standard.XXXXXX)"
trap 'rm -f "$ALL_TSV"' EXIT

echo "[bundles] discovering under $BUNDLES_ROOT"
discover_bundles_tsv "$BUNDLES_ROOT" > "$ALL_TSV"
NLINES="$(wc -l < "$ALL_TSV" | tr -d ' ')"
if [[ "$NLINES" -eq 0 ]]; then
    echo "ERROR: no bundles with .b siblings found under $BUNDLES_ROOT" >&2
    exit 4
fi
echo "[bundles] found $NLINES candidate bundle(s)"

IFS=',' read -r -a CONFIGS <<< "$CONFIGS_CSV"

# Iterate HASC bundles; for each, find ABCD partner by pair_key.
PAIRS_PROCESSED=0
while IFS=$'\t' read -r ACQ HASC_NAME HASC_DWI HASC_B HASC_MASK PAIR_KEY; do
    [[ "$ACQ" != "HASC" ]] && continue
    if [[ -n "$FILTER_RE" ]] && ! [[ "$HASC_NAME" =~ $FILTER_RE ]]; then
        continue
    fi
    ABCD_LINE="$(awk -F'\t' -v key="$PAIR_KEY" '$1 == "ABCD" && $6 == key { print; exit }' "$ALL_TSV")"
    if [[ -z "$ABCD_LINE" ]]; then
        echo "[skip] no ABCD partner for $HASC_NAME"
        continue
    fi
    IFS=$'\t' read -r _ ABCD_NAME ABCD_DWI ABCD_B ABCD_MASK _ <<< "$ABCD_LINE"

    echo "===== pair: $HASC_NAME  <->  $ABCD_NAME ====="
    PAIR_OUT="$OUT_ROOT/$HASC_NAME"
    mkdir -p "$PAIR_OUT"

    ABCD_GOLD_DIR="$PAIR_OUT/abcd_gold"
    run_mrtrix_recon "$ABCD_DWI" "$ABCD_B" "$ABCD_MASK" \
        "$ABCD_GOLD_DIR" "abcd_gold/$HASC_NAME"
    ABCD_GOLD_ODX="$ABCD_GOLD_DIR/gold.odx"

    for cfg in "${CONFIGS[@]}"; do
        CFG_BENCH_DIR="$BENCH_ROOT/$HASC_NAME/$cfg"
        COEFFS_ODX="$CFG_BENCH_DIR/coeffs.odx"
        PREDICTED_NII="$CFG_BENCH_DIR/cross/predicted.nii.gz"
        CFG_OUT="$PAIR_OUT/$cfg"

        if [[ ! -f "$COEFFS_ODX" ]]; then
            echo "[$HASC_NAME/$cfg] no coeffs.odx at $COEFFS_ODX; skipping"
            continue
        fi

        run_compare "$COEFFS_ODX" "$ABCD_GOLD_ODX" \
            "$CFG_OUT/cs_direct_vs_gold" "cs_direct_vs_gold/$HASC_NAME/$cfg"

        if [[ ! -f "$PREDICTED_NII" ]]; then
            echo "[$HASC_NAME/$cfg] no predicted.nii.gz at $PREDICTED_NII; skipping round-trip"
            continue
        fi
        run_mrtrix_recon "$PREDICTED_NII" "$ABCD_B" "$ABCD_MASK" \
            "$CFG_OUT/predicted_gold" "predicted_gold/$HASC_NAME/$cfg"
        PRED_GOLD_ODX="$CFG_OUT/predicted_gold/gold.odx"
        run_compare "$PRED_GOLD_ODX" "$ABCD_GOLD_ODX" \
            "$CFG_OUT/cs_roundtrip_vs_gold" "cs_roundtrip_vs_gold/$HASC_NAME/$cfg"
    done

    PAIRS_PROCESSED=$((PAIRS_PROCESSED + 1))
done < "$ALL_TSV"

if [[ $PAIRS_PROCESSED -eq 0 ]]; then
    echo "ERROR: no HASC↔ABCD pairs processed (filter '$FILTER_CSV'?)" >&2
    exit 4
fi
echo "[done] processed $PAIRS_PROCESSED pair(s); outputs under $OUT_ROOT"
