# Functions for the _targets pipeline (paper/paper.qmd). Every summary reads cached result CSVs from analysis/cache and
# returns a SKELETON (same columns, NA values) when the file does not exist yet, so the paper always renders and
# unfinished numbers show up as "TBD".
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(readr); library(purrr); library(stringr); library(ggplot2); library(tibble)
})

CACHE <- "analysis/cache"
MODELS3 <- c("clemson", "combined", "oxwalk")
TVS_MODELS <- c("clemson", "oxwalk", "combined", "all3",
                "orient_xyz", "orient_xyz_vm", "orient_xyz_mirror", "orient_xyz_vm_mirror", "orient_xyz_rot")
ARM_LAB <- c(clemson = "clemson (original recipe)", oxwalk = "oxwalk (original recipe)", combined = "combined (Clemson+OxWalk)",
             all3 = "all3 (Clemson+OxWalk+MAREA)",
             orient_xyz = "xyz (no VM, no augmentation)", orient_xyz_vm = "xyz + VM (no augmentation)",
             orient_xyz_mirror = "xyz + rotation/mirror aug (no VM)", orient_xyz_vm_mirror = "xyz + VM + rotation/mirror aug",
             orient_xyz_rot = "xyz + 24-rotation aug (no VM)",
             xyz = "xyz (no VM, no augmentation)", xyz_vm = "xyz + VM (no augmentation)",
             xyz_mirror = "xyz + rotation/mirror aug (no VM)", xyz_vm_mirror = "xyz + VM + rotation/mirror aug",
             xyz_rot = "xyz + 24-rotation aug (no VM)", vm = "VM only", xyz_vm_drop = "xyz+VM, xyz dropout",
             xyz_vm_rot = "xyz + VM + 24-rotation aug", with_marea = "with MAREA in training (all3 recipe)",
             no_marea = "without MAREA (combined recipe)")
lab <- function(x) unname(ifelse(x %in% names(ARM_LAB), ARM_LAB[x], x))

# ---------------------------------------------------------------------------------------------- input stamps
# A target that lists the (possibly missing) input files + their md5, so downstream targets rebuild when results arrive.
input_stamp <- function(paths) {
  paths <- paths[file.exists(paths)]
  list(files = paths, md5 = if (length(paths)) unname(tools::md5sum(paths)) else character(0))
}
glob_stamp <- function(dir, pattern) input_stamp(list.files(dir, pattern, full.names = TRUE))

# ---------------------------------------------------------------------------------------------- metrics
add_pe <- function(d) d %>% mutate(err = mass - true, pe = err / true * 100, ape = abs(pe))

part_level <- function(d, by = character(), id = "subject") {
  d %>% group_by(across(all_of(c(by, id)))) %>%
    summarise(true = sum(true), mass = sum(mass), .groups = "drop") %>% add_pe()
}
boot_mean <- function(x, n = 2000) {
  if (length(x) < 2) return(c(NA_real_, NA_real_))
  m <- replicate(n, mean(sample(x, length(x), TRUE)))
  unname(quantile(m, c(.025, .975)))
}
summ_metric <- function(p, by) {
  p %>% group_by(across(all_of(by))) %>% group_modify(~ {
    set.seed(1); ci <- boot_mean(.x$ape)
    tibble(n = nrow(.x), MAPE = mean(.x$ape), SD = sd(.x$ape), MAPE_lo = ci[1], MAPE_hi = ci[2],
           MPE = mean(.x$pe), bias_steps = mean(.x$err))
  }) %>% ungroup()
}
ci_str <- function(m, lo, hi, d = 1) ifelse(is.na(m), "TBD", sprintf(paste0("%.", d, "f (%.", d, "f, %.", d, "f)"), m, lo, hi))
f1 <- function(x, d = 1) ifelse(is.na(x), "TBD", formatC(x, format = "f", digits = d))

subset_defs <- function(d) {
  run <- str_detect(d$activity, "run")
  bind_rows(
    d %>% filter(cohort == "clemson") %>% mutate(subset = "Clemson wrist"),
    d %>% filter(cohort == "oxwalk") %>% mutate(subset = "OxWalk wrist"),
    d %>% filter(cohort == "marea", !run) %>% mutate(subset = "MAREA walking"),
    d %>% filter(cohort == "marea", run) %>% mutate(subset = "MAREA running"),
    d %>% filter(cohort == "marea") %>% mutate(subset = "MAREA walking + running"),
    d %>% mutate(subset = "All cohorts"))
}
SUBSETS <- c("Clemson wrist", "OxWalk wrist", "MAREA walking", "MAREA running", "MAREA walking + running", "All cohorts")

# ---------------------------------------------------------------------------------------------- MAREA (original models)
load_marea <- function(stamp) {
  if (!length(stamp$files)) return(NULL)
  read_csv(stamp$files[1], show_col_types = FALSE) %>%
    mutate(kind = if_else(str_detect(activity, "run"), "run", "walk"), subject = as.integer(subject),
           cohort = "marea") %>% add_pe()
}

tab_marea <- function(d) {
  if (is.null(d)) return(tibble(model = MODELS3, scope = NA_character_, convention = NA_character_, n = NA_integer_,
                                MAPE = NA_real_, MAPE_lo = NA_real_, MAPE_hi = NA_real_, MPE = NA_real_, bias_steps = NA_real_))
  koff <- function(dd, scope) dd %>% part_level("model") %>% summ_metric("model") %>%
    mutate(scope = scope, convention = "per participant (sum over recordings)")
  rec <- function(dd, scope) dd %>% group_by(model) %>%
    summarise(n = n(), MAPE = mean(ape), SD = sd(ape), MPE = mean(pe), bias_steps = mean(err), .groups = "drop") %>%
    mutate(scope = scope, convention = "per recording", MAPE_lo = NA_real_, MAPE_hi = NA_real_)
  bind_rows(koff(filter(d, kind == "walk"), "walking"), koff(filter(d, kind == "run"), "running"), koff(d, "all activities"),
            rec(filter(d, kind == "walk"), "walking"), rec(filter(d, kind == "run"), "running"), rec(d, "all activities")) %>%
    mutate(scope = factor(scope, c("walking", "running", "all activities"))) %>% arrange(convention, scope, model)
}

tab_running_share <- function(d) {
  if (is.null(d)) return(tibble(model = MODELS3, share_steps_running = NA_real_, missed_steps_total = NA_real_,
                                missed_steps_running = NA_real_, share_missed_from_running = NA_real_))
  d %>% group_by(model) %>% summarise(
    share_steps_running = 100 * sum(true[kind == "run"]) / sum(true),
    missed_steps_total = sum(true - mass), missed_steps_running = sum((true - mass)[kind == "run"]),
    share_missed_from_running = 100 * missed_steps_running / missed_steps_total, .groups = "drop")
}

# ---------------------------------------------------------------------------------------------- published numbers
published_sensors <- tribble(
  ~setting, ~data, ~MAPE, ~MPE,
  "In-domain, held-out", "Clemson wrist, regular", 4.61, -2.33,
  "In-domain, held-out", "Clemson wrist, semi-regular", 9.90, -5.28,
  "In-domain, held-out", "Clemson wrist, irregular", 15.78, 2.87,
  "In-domain, held-out", "OxWalk wrist (free-living)", 13.06, -2.78,
  "Zero-shot OxWalk -> Clemson", "regular", 12.97, -7.84,
  "Zero-shot OxWalk -> Clemson", "semi-regular", 13.04, -8.68,
  "Zero-shot OxWalk -> Clemson", "irregular", 16.56, -1.11,
  "Zero-shot Clemson -> OxWalk", "OxWalk (participants)", 28.75, 10.89)

published_koffman <- tribble(   # Koffman et al., MAREA walking: MAPE (SD), bias in steps (SD), second-level recall, F1
  ~method, ~MAPE, ~MAPE_sd, ~bias, ~bias_sd, ~recall, ~F1,
  "stepcount SSL", 11.4, 10, -210, 262, .99, .99, "stepcount RF", 12.9, 10, -232, 277, .98, .99,
  "SDT", 16.1, 11, -257, 273, .99, .99, "Verisense revised", 19.2, 11, -333, 324, .94, .97,
  "Verisense original", 30.6, 15, -491, 445, .86, .92, "ADEPT", 36.2, 23, -652, 614, .64, .75,
  "ActiLife", 39.0, 13, -633, 514, .92, .96, "Oak", 42.5, 28, -742, 730, .58, .69)

tab_koffman <- function(d) {
  pub <- published_koffman %>% transmute(method = paste(method, "(published)"),
                                         MAPE = sprintf("%.1f (%.0f)", MAPE, MAPE_sd), bias = sprintf("%.0f (%.0f)", bias, bias_sd),
                                         recall = sprintf("%.2f", recall), F1 = sprintf("%.2f", F1))
  if (is.null(d)) ours <- tibble(method = paste(MODELS3, "(this work)"), MAPE = "TBD", bias = "TBD", recall = "n/a", F1 = "n/a")
  else {
    p <- d %>% filter(kind == "walk") %>% part_level("model")
    ours <- p %>% group_by(model) %>% summarise(MAPE = sprintf("%.1f (%.0f)", mean(ape), sd(ape)),
                                                bias = sprintf("%.0f (%.0f)", mean(err), sd(err)), .groups = "drop") %>%
      transmute(method = paste(model, "(this work)"), MAPE, bias, recall = "n/a", F1 = "n/a")
  }
  bind_rows(pub, ours)
}

# ---------------------------------------------------------------------------------------------- CV with MAREA
load_cv <- function(stamp) {
  if (!length(stamp$files)) return(NULL)
  map_dfr(stamp$files, read_csv, show_col_types = FALSE) %>% add_pe()
}
tab_cv <- function(d) {
  if (is.null(d)) return(expand_grid(arm = c("no_marea", "with_marea"), subset = SUBSETS[1:5]) %>%
                           mutate(n = NA_integer_, MAPE = NA_real_, MAPE_lo = NA_real_, MAPE_hi = NA_real_, MPE = NA_real_, bias_steps = NA_real_))
  subset_defs(d) %>% filter(subset != "All cohorts") %>% part_level(c("arm", "subset")) %>% summ_metric(c("arm", "subset"))
}
fig_cv <- function(d) {
  if (is.null(d)) return(placeholder_plot("Held-out CV results not found.\nRun scripts/slurm/cv_marea.sbatch"))
  p <- subset_defs(d) %>% filter(subset != "All cohorts") %>% part_level(c("arm", "subset")) %>%
    mutate(subset = factor(subset, SUBSETS), arm = lab(arm))
  ggplot(p, aes(subset, ape, colour = arm)) +
    geom_point(position = position_jitterdodge(jitter.width = .15, dodge.width = .6), alpha = .55, size = 1.6) +
    stat_summary(aes(group = arm), fun = mean, geom = "crossbar", width = .5, fatten = 1.2, position = position_dodge(.6), colour = "black") +
    scale_y_continuous("Absolute percentage error per held-out participant (%)", trans = "sqrt", breaks = c(0, 2, 5, 10, 25, 50, 80)) +
    scale_colour_manual(values = c("#1b6ca8", "#d1495b")) + labs(x = NULL, colour = NULL) +
    theme_minimal(base_size = 10) + theme(legend.position = "top", axis.text.x = element_text(angle = 20, hjust = 1))
}

fig_marea <- function(d) {
  if (is.null(d)) return(placeholder_plot("MAREA validation cache not found.\nRun python analysis/marea_eval.py"))
  ord <- c("treadmill_walk", "treadmill_slopewalk", "indoor_walk", "outdoor_walk", "treadmill_run", "indoor_run", "outdoor_run")
  ggplot(d %>% mutate(activity = factor(activity, ord)), aes(activity, pe, colour = model)) +
    geom_hline(yintercept = 0, colour = "grey60") +
    geom_boxplot(outlier.size = .8, linewidth = .4, width = .7, position = position_dodge(.8)) +
    scale_colour_manual(values = c(clemson = "#1b6ca8", combined = "#2a9d8f", oxwalk = "#d1495b")) +
    labs(x = NULL, y = "Percentage error per recording (%)", colour = NULL) +
    theme_minimal(base_size = 10) + theme(legend.position = "top", axis.text.x = element_text(angle = 25, hjust = 1))
}

# ---------------------------------------------------------------------------------------------- Mobilise-D TVS
load_tvs_manifest <- function(stamp) {
  if (!length(stamp$files)) return(NULL)
  read_csv(stamp$files[1], show_col_types = FALSE)
}
tab_tvs_desc <- function(m) {
  if (is.null(m)) return(tibble(cohort = c("CHF", "COPD", "HA", "MS", "PD", "PFF"), participants = NA, left_wrist = NA,
                                right_wrist = NA, steps = NA, recordings = NA, median_spm = NA))
  l <- m %>% filter(is_last_trial, n_steps > 0)
  bind_rows(
    l %>% group_by(cohort) %>% summarise(participants = n_distinct(id), left_wrist = n_distinct(id[wrist == "left"]),
                                         right_wrist = n_distinct(id[wrist == "right"]), steps = sum(n_steps),
                                         recordings = n(), median_spm = median(steps_per_min), .groups = "drop"),
    l %>% summarise(cohort = "All", participants = n_distinct(id), left_wrist = n_distinct(id[wrist == "left"]),
                    right_wrist = n_distinct(id[wrist == "right"]), steps = sum(n_steps), recordings = n(),
                    median_spm = median(steps_per_min)))
}

load_tvs <- function(stamp) {
  f <- stamp$files[basename(stamp$files) == "orient_tvs_participants.csv"]
  if (!length(f)) return(NULL)
  read_csv(f, show_col_types = FALSE)
}
tvs_part <- function(p, count = "bout") p %>% mutate(mass = .data[[count]], subject = id) %>% add_pe()
tab_tvs <- function(p, by = "model", count = "bout", models = TVS_MODELS) {
  if (is.null(p)) {
    sk <- tibble(model = models)
    if ("wrist" %in% by) sk <- expand_grid(sk, wrist = c("left", "right"))
    return(sk %>% mutate(n = NA_integer_, MAPE = NA_real_, MAPE_lo = NA_real_, MAPE_hi = NA_real_, MPE = NA_real_, bias_steps = NA_real_))
  }
  tvs_part(p, count) %>% summ_metric(by) %>%
    mutate(model = factor(model, TVS_MODELS)) %>% arrange(model) %>% mutate(model = as.character(model))
}
fig_tvs <- function(p) {
  if (is.null(p)) return(placeholder_plot("Mobilise-D TVS scoring not found.\nRun scripts/slurm/run_orient_pipeline.sh (stage 2)"))
  q <- tvs_part(p) %>% mutate(model = factor(lab(model), unique(lab(TVS_MODELS))), side = str_to_title(wrist))
  ggplot(q, aes(model, ape)) +
    geom_boxplot(outlier.shape = NA, linewidth = .4, colour = "grey30") +
    geom_jitter(aes(colour = side), width = .15, alpha = .55, size = 1.3) +
    coord_flip(ylim = c(0, 100)) + scale_colour_manual(values = c(Left = "#1b6ca8", Right = "#d1495b")) +
    labs(x = NULL, y = "Absolute percentage error per participant, inside walking bouts (%)", colour = "wrist") +
    theme_minimal(base_size = 10) + theme(legend.position = "top")
}

# ---------------------------------------------------------------------------------------------- orientation arms CV / LOCO
load_orient <- function(stamp, protocol) {
  f <- stamp$files[str_starts(basename(stamp$files), paste0(protocol, "_"))]
  if (!length(f)) return(NULL)
  map_dfr(f, read_csv, show_col_types = FALSE) %>% filter(mode %in% c("full", "vm_only")) %>%
    mutate(arm = if_else(mode == "vm_only", paste0(arm, "/vm_only"), arm)) %>% add_pe()
}
tab_orient <- function(d, arms = c("xyz", "xyz_vm", "xyz_mirror", "xyz_vm_mirror")) {
  sk <- expand_grid(arm = arms, subset = SUBSETS) %>%
    mutate(as_recorded = NA_real_, mean_24rot = NA_real_, worst_24rot = NA_real_, mean_mirror = NA_real_)
  if (is.null(d)) return(sk)
  per_rot <- subset_defs(d) %>% part_level(c("arm", "subset", "rot")) %>%
    group_by(arm, subset, rot) %>% summarise(MAPE = mean(ape), .groups = "drop")
  per_rot %>% group_by(arm, subset) %>% summarise(
    as_recorded = MAPE[rot == 0][1], mean_24rot = mean(MAPE[rot < 24]), worst_24rot = max(MAPE[rot < 24]),
    mean_mirror = if (any(rot >= 24)) mean(MAPE[rot >= 24]) else NA_real_, .groups = "drop") %>%
    mutate(arm = factor(arm, unique(c(arms, arm))), subset = factor(subset, SUBSETS)) %>% arrange(arm, subset) %>%
    mutate(arm = as.character(arm), subset = as.character(subset))
}
fig_orient <- function(d, title) {
  if (is.null(d)) return(placeholder_plot(paste0(title, " results not found.\nRun scripts/slurm/run_orient_pipeline.sh")))
  per_rot <- subset_defs(d) %>% filter(subset == "All cohorts") %>% part_level(c("arm", "rot")) %>%
    group_by(arm, rot) %>% summarise(MAPE = mean(ape), .groups = "drop") %>% mutate(arm = lab(arm), mirrored = rot >= 24)
  ggplot(per_rot, aes(rot, MAPE, colour = arm)) +
    geom_vline(xintercept = 23.5, colour = "grey75", linetype = 2) +
    geom_line(linewidth = .5, alpha = .7) + geom_point(aes(shape = mirrored), size = 1.6) +
    labs(x = "axis rotation applied to the held-out signal (0 = as recorded; 24+ = mirrored)", y = "MAPE per participant (%)",
         colour = NULL, shape = "mirrored") + theme_minimal(base_size = 10) + theme(legend.position = "top")
}

# ---------------------------------------------------------------------------------------------- misc
placeholder_plot <- function(msg) {
  ggplot() + annotate("text", x = 0, y = 0, label = paste0("PENDING\n", msg), size = 4, colour = "grey35", lineheight = 1.1) +
    xlim(-1, 1) + ylim(-1, 1) + theme_void() +
    theme(panel.border = element_rect(colour = "grey70", fill = NA, linetype = 2))
}

tab_status <- function(...) {
  s <- list(...)
  tibble(
    result = c("MAREA validation of clemson / oxwalk / combined", "Koffman et al. per-participant totals", "Held-out CV with MAREA (all3 recipe)",
               "Mobilise-D TVS conversion (manifest)", "TVS scoring of all models", "Orientation arms: 5-fold CV", "Orientation arms: leave-one-cohort-out"),
    produced_by = c("python analysis/marea_eval.py", "Rscript analysis/fetch_koffman_marea.R", "scripts/slurm/cv_marea.sbatch",
                    "python scripts/prepare_mobilised_tvs.py", "scripts/slurm/run_orient_pipeline.sh (stage 2)",
                    "run_orient_pipeline.sh (stage 3)", "run_orient_pipeline.sh (stage 4)"),
    files_found = map_int(s, ~ length(.x$files)),
    status = if_else(files_found > 0, "available", "pending"))
}

# ---------------------------------------------------------------------------------------------- scalar values for inline prose
pick <- function(tab, ..., col) {
  q <- enquos(...)
  r <- tab
  for (e in q) r <- filter(r, !!e)
  if (!nrow(r) || !col %in% names(r)) NA_real_ else r[[col]][1]
}
make_vals <- function(marea, tab_m, share, cv, tvs_desc, tvs_t, tvs_side, orient_cv_t, orient_loco_t) {
  v <- list()
  v$marea_n_subjects <- if (is.null(marea)) NA else n_distinct(marea$subject)
  v$marea_n_rec <- if (is.null(marea)) NA else nrow(marea) / n_distinct(marea$model)
  v$marea_steps <- if (is.null(marea)) NA else sum(marea$true) / n_distinct(marea$model)
  for (m in MODELS3) {
    v[[paste0("walk_mape_", m)]] <- pick(tab_m, model == m, scope == "walking", convention == "per participant (sum over recordings)", col = "MAPE")
    v[[paste0("walk_mpe_", m)]] <- pick(tab_m, model == m, scope == "walking", convention == "per participant (sum over recordings)", col = "MPE")
    v[[paste0("run_mape_", m)]] <- pick(tab_m, model == m, scope == "running", convention == "per participant (sum over recordings)", col = "MAPE")
    v[[paste0("run_mpe_", m)]] <- pick(tab_m, model == m, scope == "running", convention == "per participant (sum over recordings)", col = "MPE")
    v[[paste0("all_mape_", m)]] <- pick(tab_m, model == m, scope == "all activities", convention == "per participant (sum over recordings)", col = "MAPE")
    v[[paste0("run_share_missed_", m)]] <- pick(share, model == m, col = "share_missed_from_running")
  }
  rng <- function(pat) { x <- unlist(v[paste0(pat, MODELS3)]); if (all(is.na(x))) c(NA_real_, NA_real_) else range(x, na.rm = TRUE) }
  v$walk_lo <- rng("walk_mape_")[1]; v$walk_hi <- rng("walk_mape_")[2]
  v$run_lo <- rng("run_mape_")[1]; v$run_hi <- rng("run_mape_")[2]
  v$share_missed_lo <- rng("run_share_missed_")[1]; v$share_missed_hi <- rng("run_share_missed_")[2]
  v$share_steps_running <- pick(share, model == "combined", col = "share_steps_running")
  for (a in c("no_marea", "with_marea")) for (s in SUBSETS[1:5]) {
    key <- paste0("cv_", a, "_", str_replace_all(tolower(s), "[^a-z]+", "_"))
    v[[paste0(key, "_mape")]] <- pick(cv, arm == a, subset == s, col = "MAPE")
    v[[paste0(key, "_mpe")]] <- pick(cv, arm == a, subset == s, col = "MPE")
  }
  v$tvs_participants <- pick(tvs_desc, cohort == "All", col = "participants")
  v$tvs_left <- pick(tvs_desc, cohort == "All", col = "left_wrist")
  v$tvs_right <- pick(tvs_desc, cohort == "All", col = "right_wrist")
  v$tvs_steps <- pick(tvs_desc, cohort == "All", col = "steps")
  v$tvs_recordings <- pick(tvs_desc, cohort == "All", col = "recordings")
  v$tvs_median_spm <- pick(tvs_desc, cohort == "All", col = "median_spm")
  for (m in TVS_MODELS) {
    v[[paste0("tvs_mape_", m)]] <- pick(tvs_t, model == m, col = "MAPE")
    v[[paste0("tvs_mpe_", m)]] <- pick(tvs_t, model == m, col = "MPE")
    v[[paste0("tvs_left_mape_", m)]] <- pick(tvs_side, model == m, wrist == "left", col = "MAPE")
    v[[paste0("tvs_right_mape_", m)]] <- pick(tvs_side, model == m, wrist == "right", col = "MAPE")
  }
  v
}
