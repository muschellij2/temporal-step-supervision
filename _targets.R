# Pipeline for paper/paper.qmd:   Rscript -e 'targets::tar_make()'   (from the repo root)
# Reads cached results in analysis/cache (produced by the Python/SLURM scripts). Anything that does not exist yet becomes
# a skeleton with NA ("TBD" in the paper). Re-running after new results are copied in rebuilds only what changed.
library(targets)
library(tarchetypes)
tar_option_set(packages = c("dplyr", "tidyr", "readr", "purrr", "stringr", "ggplot2", "tibble"))
tar_source("R")

# Quarto ships with RStudio on this machine; tell tar_quarto where it is if `quarto` is not on PATH.
rs <- "/Applications/RStudio.app/Contents/Resources/app/quarto/bin"
if (!nzchar(Sys.which("quarto")) && dir.exists(rs)) Sys.setenv(QUARTO_PATH = file.path(rs, "quarto"), PATH = paste(rs, Sys.getenv("PATH"), sep = ":"))

list(
  # ---- which inputs exist (always re-checked; downstream targets rebuild only when the md5 changes)
  tar_target(stamp_marea,    input_stamp(file.path(CACHE, "marea_counts.csv")), cue = tar_cue("always")),
  tar_target(stamp_koffman,  input_stamp(file.path(CACHE, "koffman_marea_totals.csv")), cue = tar_cue("always")),
  tar_target(stamp_cv,       glob_stamp(file.path(CACHE, "cv_marea"), "\\.csv$"), cue = tar_cue("always")),
  tar_target(stamp_manifest, input_stamp("data/mobilised_tvs/manifest.csv"), cue = tar_cue("always")),
  tar_target(stamp_tvs,      input_stamp(file.path(CACHE, c("orient_tvs_participants.csv", "orient_tvs_recordings.csv"))), cue = tar_cue("always")),
  tar_target(stamp_orient,   glob_stamp(file.path(CACHE, "orient"), "\\.csv$"), cue = tar_cue("always")),
  tar_target(stamp_cohort,   input_stamp(file.path(CACHE, "cohort_protocols.csv")), cue = tar_cue("always")),
  tar_target(stamp_fuller,   glob_stamp(file.path(CACHE, "orient_fuller"), "^cv_.*\\.csv$"), cue = tar_cue("always")),
  tar_target(stamp_rot,      glob_stamp(CACHE, "^orient_tvs_rotations_.*\\.csv$"), cue = tar_cue("always")),

  # ---- MAREA: original clemson / oxwalk / combined models, as-is
  tar_target(marea,          load_marea(stamp_marea)),
  tar_target(tab_marea_main, tab_marea(marea)),
  tar_target(tab_run_share,  tab_running_share(marea)),
  tar_target(tab_koff,       tab_koffman(marea)),
  tar_target(fig_marea_pe,   fig_marea(marea)),

  # ---- held-out CV with MAREA in training (all3 recipe) vs without (combined recipe)
  tar_target(cv,             load_cv(stamp_cv)),
  tar_target(tab_cv_main,    tab_cv(cv)),
  tar_target(fig_cv_main,    fig_cv(cv)),

  # ---- Mobilise-D TVS
  tar_target(manifest,       load_tvs_manifest(stamp_manifest)),
  tar_target(tab_tvs_description, tab_tvs_desc(manifest)),
  tar_target(tvs,            load_tvs(stamp_tvs)),
  tar_target(tab_tvs_models, tab_tvs(tvs)),
  tar_target(tab_tvs_all,    tab_tvs(tvs, count = "all")),
  tar_target(tab_tvs_wrist,  tab_tvs(tvs, by = c("model", "wrist"))),
  tar_target(fig_tvs_main,   fig_tvs(tvs)),

  # ---- effect of augmentation on TVS (paired) and oracle axis-swap sweep (all 48 signed permutations)
  tar_target(tab_aug_tvs,    tab_aug_effect(tvs)),
  tar_target(rot,            load_rotations(stamp_rot)),
  tar_target(tab_oracle_main, tab_oracle(rot)),
  tar_target(fig_oracle_main, fig_oracle(rot)),
  tar_target(tab_tta_tvs_main, tab_tta_tvs(rot)),
  tar_target(tab_subset_tvs_main, tab_subset_tvs(rot)),
  tar_target(tab_subset_cv_main,  tab_subset_cv(orient_cv)),
  tar_target(tab_tta_cv,     tab_tta(orient_cv)),
  tar_target(tab_tta_loco,   tab_tta(orient_loco)),

  # ---- orientation / VM / augmentation arms: CV and leave-one-cohort-out, each held-out signal scored under 24 rotations
  tar_target(orient_cv,      load_orient(stamp_orient, "cv")),
  tar_target(orient_loco,    load_orient(stamp_orient, "loco")),
  tar_target(tab_orient_cv,  tab_orient(orient_cv)),
  tar_target(tab_orient_loco, tab_orient(orient_loco)),
  tar_target(fig_orient_cv,  fig_orient(orient_cv, "Orientation CV")),
  tar_target(fig_orient_loco, fig_orient(orient_loco, "Orientation leave-one-cohort-out")),

  # ---- cohort protocols and the fuller 4-cohort models (place8 / place16, trained on Clemson + OxWalk + MAREA + TVS)
  tar_target(cohort_stats,   load_cohort_stats(stamp_cohort)),
  tar_target(tab_cohort,     tab_cohort_numbers(cohort_stats)),
  tar_target(fuller,         load_fuller(stamp_fuller)),
  tar_target(tab_fuller_main, tab_fuller(fuller)),
  tar_target(fig_fuller_main, fig_fuller(fuller)),

  # ---- bookkeeping + scalars for inline prose
  tar_target(tab_status_main, tab_status(stamp_marea, stamp_koffman, stamp_cv, stamp_manifest, stamp_tvs,
                                         glob_stamp(file.path(CACHE, "orient"), "^cv_.*\\.csv$"),
                                         glob_stamp(file.path(CACHE, "orient"), "^loco_.*\\.csv$"),
                                         glob_stamp(CACHE, "^orient_tvs_rotations_.*\\.csv$"), stamp_cohort, stamp_fuller)),
  tar_target(vals, make_vals(marea, tab_marea_main, tab_run_share, tab_cv_main, tab_tvs_description,
                             tab_tvs_models, tab_tvs_wrist, tab_orient_cv, tab_orient_loco)),

  # ---- the paper (PDF via Typst; tar_quarto tracks the tar_read() calls inside the .qmd)
  tar_quarto(paper, "paper.qmd", quiet = FALSE)
)
