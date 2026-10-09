#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Regenerate docs/cli/*.md from the tools' --help output.

Usage: python docs/tools/make_cli_pages.py [path/to/target/release]

CI runs this after building and fails if the committed pages differ.
"""

import subprocess
import sys
from pathlib import Path

TOOLS = {
    "cs-qc": "Image-quality metrics for a DWI series, written as JSON and/or a one-row TSV with a JSON data "
             "dictionary. See {doc}`../user/qc`.",
    "cs-fit": "Fits a 3D-SHORE basis to every voxel in the mask and writes the coefficients with a JSON sidecar; "
              "optionally also an ODX file. See {doc}`../user/shore`.",
    "cs-odf": "Computes ODF spherical-harmonic coefficients, peaks and scalars from `cs-fit` coefficients and "
              "writes an ODX file. See {doc}`../user/shore`.",
    "cs-synth": "Synthesizes a DWI series for a target gradient table from `cs-fit` coefficients. "
                "See {doc}`../user/shore`.",
    "cs-dti": "Fits the diffusion tensor with RESTORE. See {doc}`../user/dti`.",
    "cs-response": "Estimates white-matter, grey-matter and cerebrospinal-fluid response functions from "
                   "single-shell data. See {doc}`../user/multitissue`.",
    "cs-ss3t": "Single-shell three-tissue CSD with given response functions. See {doc}`../user/multitissue`.",
    "cs-mtnorm": "Multi-tissue intensity normalisation of tissue maps. See {doc}`../user/multitissue`.",
    "cs-ss3t-full": "Response estimation, SS3T-CSD and intensity normalisation in one step. "
                    "See {doc}`../user/multitissue`.",
}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    bindir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "target" / "release"
    out = root / "docs" / "cli"
    out.mkdir(exist_ok=True)
    index = ["# Command-line reference", "", "Each page reproduces the tool's `--help` output.", "",
             "| Tool | Purpose |", "|---|---|"]
    for tool, blurb in TOOLS.items():
        text = subprocess.run([str(bindir / tool), "--help"], check=True, capture_output=True,
                              text=True).stdout.rstrip()
        page = [f"# {tool}", "", blurb, "", "```text", text, "```", ""]
        (out / f"{tool}.md").write_text("\n".join(page))
        index.append(f"| [`{tool}`]({tool}.md) | {blurb.split('. See')[0].rstrip('.')}. |")
    index += ["", "```{toctree}", ":hidden:", "", *TOOLS, "```", ""]
    (out / "index.md").write_text("\n".join(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
