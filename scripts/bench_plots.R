#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT OR Apache-2.0
# Visualize cs-dmri fit-option benchmark results.
#
# Reads <bench-root>/summary.csv and writes a panel of PNGs to
# <bench-root>/plots/. Default root is ~/cs-bench-csdsi/focused.
#
# All R²/RMSE/sparsity aggregates are MEDIANS — robust to the tail of
# low-variance/edge voxels that drag the mean.

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(stringr)
  library(forcats); library(ggplot2); library(scales); library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else file.path(Sys.getenv("HOME"), "cs-bench-csdsi", "focused")
out_dir <- file.path(root, "plots")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

raw <- read_csv(file.path(root, "summary.csv"), show_col_types = FALSE)

# Subject/acq parsing — bundle is sub-XXXX_ses-Y_acq-ZZ[_run-N]
df <- raw |>
  mutate(
    subject = str_extract(bundle, "sub-[A-Za-z0-9]+"),
    acq     = str_extract(bundle, "(?<=acq-)[A-Za-z0-9]+"),
    acq     = if_else(str_starts(acq, "HASC"), "HASC55", acq),
    config  = factor(name, levels = c("l2_baseline", "l1_ratio_1e-3",
                                      "l1_pathbic_n20", "l1_pathbic_ro4",
                                      "l1_pathbic_nn"))
  )

# Cross rows = HASC bundles with a paired ABCD partner.
cross <- df |> filter(!is.na(cross_r2_p50))

cfg_colors <- c(
  "l2_baseline"    = "#888888",
  "l1_ratio_1e-3"  = "#d95f02",
  "l1_pathbic_n20" = "#1b9e77",
  "l1_pathbic_ro4" = "#7570b3",
  "l1_pathbic_nn"  = "#e7298a"
)
theme_bench <- theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "right",
        panel.grid.minor = element_blank())

save_plot <- function(p, name, w = 8, h = 5) {
  path <- file.path(out_dir, paste0(name, ".png"))
  ggsave(path, p, width = w, height = h, dpi = 150)
  cat("wrote", path, "\n")
}

# ---- 1. In-sample R² vs cross R² (does fit quality transfer?) -------------
p1 <- ggplot(cross, aes(r2_p50, cross_r2_p50, color = config)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey60") +
  geom_point(aes(shape = subject), size = 3.2, alpha = 0.85) +
  scale_color_manual(values = cfg_colors) +
  scale_x_continuous(limits = c(0.93, 1.0), breaks = pretty_breaks(5)) +
  scale_y_continuous(limits = c(0.90, 1.0), breaks = pretty_breaks(5)) +
  labs(
    title = "In-sample fit quality vs HASC55 → ABCD generalization (medians)",
    subtitle = "Each point: one (subject × config). Distance below the dashed line = generalization gap.",
    x = "in-sample R² median (HASC fit)",
    y = "cross R² median (ABCD predicted vs measured)",
    color = "config", shape = "subject"
  ) +
  theme_bench
save_plot(p1, "01_insample_vs_cross_r2", w = 9, h = 5.5)

# ---- 2. Per-config distribution of cross R² across subjects ---------------
p2 <- ggplot(cross, aes(fct_rev(config), cross_r2_p50, fill = config)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5, width = 0.6) +
  geom_jitter(aes(color = subject), width = 0.12, size = 2.6, alpha = 0.9) +
  scale_fill_manual(values = cfg_colors, guide = "none") +
  scale_color_brewer(palette = "Dark2") +
  coord_flip() +
  scale_y_continuous(limits = c(0.90, 1.0), breaks = pretty_breaks(6)) +
  labs(
    title = "Cross R² median by config, across subjects",
    subtitle = "Spread = config robustness across subjects.",
    x = NULL, y = "cross R² (median over masked voxels)"
  ) +
  theme_bench
save_plot(p2, "02_cross_r2_by_config", w = 8, h = 5)

# ---- 3. Per-method: in-sample vs cross R² (paired by subject) -------------
paired <- cross |>
  select(subject, config, `in-sample (HASC)` = r2_p50,
         `cross (ABCD)` = cross_r2_p50) |>
  pivot_longer(c(`in-sample (HASC)`, `cross (ABCD)`),
               names_to = "regime", values_to = "r2") |>
  mutate(regime = factor(regime, levels = c("in-sample (HASC)", "cross (ABCD)")))

p3 <- ggplot(paired, aes(regime, r2)) +
  geom_line(aes(group = subject, color = subject), alpha = 0.6, linewidth = 0.6) +
  geom_point(aes(color = subject, shape = subject), size = 3) +
  facet_wrap(~ config, nrow = 1) +
  scale_color_brewer(palette = "Dark2") +
  scale_y_continuous(limits = c(0.90, 1.0), breaks = pretty_breaks(6)) +
  labs(
    title = "Per fitting method: in-sample vs cross-acquisition R² (medians)",
    subtitle = "Each line: one subject. Slope = generalization cost from HASC to ABCD for that method.",
    x = NULL, y = "R² (median over masked voxels)",
    color = "subject", shape = "subject"
  ) +
  theme_bench +
  theme(axis.text.x = element_text(angle = 25, hjust = 1))
save_plot(p3, "03_per_method_insample_vs_cross", w = 12, h = 5)

# ---- 4. Wall time vs cross R² (cost/quality frontier) ---------------------
# Only renders if fit_wall_s is populated; the bench script doesn't record
# wall time when cs-fit is skipped (cached run), so re-running the script
# without --force loses these.
if (any(is.finite(cross$fit_wall_s))) {
  p4 <- ggplot(cross, aes(fit_wall_s, cross_r2_p50, color = config)) +
    geom_point(aes(shape = subject), size = 3.2, alpha = 0.85) +
    scale_color_manual(values = cfg_colors) +
    scale_x_log10(breaks = c(2, 5, 10, 30, 60, 120, 300, 600, 1200),
                  labels = label_number(accuracy = 1)) +
    scale_y_continuous(limits = c(0.90, 1.0)) +
    labs(
      title = "Cost / quality frontier: fit wall time vs cross R² median",
      subtitle = "Look for configs that climb the y-axis without sliding right.",
      x = "fit wall (s, log)", y = "cross R² median",
      color = "config", shape = "subject"
    ) +
    theme_bench
  save_plot(p4, "04_walltime_vs_cross_r2", w = 9, h = 5.5)
} else {
  message("skipping plot 4: fit_wall_s is unset (cs-fit was cached-skipped on the last bench run)")
}

# ---- 5. Cross R² vs coherence_index (does fixel coherence predict accuracy?)
p5 <- ggplot(cross, aes(coherence_index, cross_r2_p50, color = config)) +
  geom_point(aes(shape = subject), size = 3.2, alpha = 0.85) +
  scale_color_manual(values = cfg_colors) +
  scale_y_continuous(limits = c(0.90, 1.0)) +
  labs(
    title = "Fixel coherence vs HASC→ABCD prediction quality (median)",
    subtitle = "Coherence rewards aligned neighbors; cross R² rewards signal accuracy. Different axes.",
    x = "coherence_index (qa-weighted, on HASC ODX)",
    y = "cross R² median",
    color = "config", shape = "subject"
  ) +
  theme_bench
save_plot(p5, "05_coherence_vs_cross_r2", w = 9, h = 5.5)

# ---- 6. Sparsity vs in-sample R² (regularization tradeoff) ----------------
p6 <- ggplot(df, aes(sparsity_p50, r2_p50, color = config)) +
  geom_point(aes(shape = acq), size = 3, alpha = 0.85) +
  scale_color_manual(values = cfg_colors) +
  scale_x_continuous(limits = c(0, 1.05)) +
  labs(
    title = "Sparsity vs in-sample R² (medians)",
    subtitle = "L2 baseline pegs at 1.0 (dense). L1 paths trade coefficients for fit.",
    x = "sparsity (median fraction of nonzero coefficients)",
    y = "in-sample R² median",
    color = "config", shape = "acq"
  ) +
  theme_bench
save_plot(p6, "06_sparsity_vs_r2", w = 9, h = 5.5)

# ---- 7. Generalization gap per (subject, config) --------------------------
gap <- cross |>
  mutate(gap = r2_p50 - cross_r2_p50)

p7 <- ggplot(gap, aes(fct_rev(config), gap, fill = config)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
  facet_wrap(~ subject, ncol = 2) +
  scale_fill_manual(values = cfg_colors, guide = "none") +
  coord_flip() +
  labs(
    title = "Generalization gap: in-sample R² median − cross R² median",
    subtitle = "Larger bars = more overfitting to HASC's q-space sampling.",
    x = NULL, y = "in-sample − cross  (R² units, medians)"
  ) +
  theme_bench
save_plot(p7, "07_generalization_gap", w = 9, h = 6)

# ---- 8. Composite landing page --------------------------------------------
landing <- (p1 | p2) / p3 +
  plot_annotation(
    title = "cs-dmri fit-option sweep — HASC55 → ABCD cross-prediction (medians)",
    subtitle = sprintf("source: %s", root),
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )
save_plot(landing, "00_landing", w = 16, h = 11)

cat("\nAll plots written under:", out_dir, "\n")
