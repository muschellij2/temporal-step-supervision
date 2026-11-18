# Downloads the per-participant/per-activity MAREA totals published with
# Koffman & Muschelli (JMPB 2024) from https://github.com/lilykoff/step_algorithms
# (results/all_algorithms/total_steps_bysubject.rds) and writes analysis/cache/koffman_marea_totals.csv.
# These are derived step totals only (no raw MAREA data). Walking recordings only (running was removed upstream).
# Run from the repo root:  Rscript analysis/fetch_koffman_marea.R
suppressMessages(library(dplyr))
url <- "https://raw.githubusercontent.com/lilykoff/step_algorithms/main/results/all_algorithms/total_steps_bysubject.rds"
f <- tempfile(fileext = ".rds"); download.file(url, f, mode = "wb", quiet = TRUE)
t <- readRDS(f) %>% filter(id_study == "marea")
dir.create("analysis/cache", showWarnings = FALSE)
write.csv(t, "analysis/cache/koffman_marea_totals.csv", row.names = FALSE)
cat("wrote", nrow(t), "rows\n")
