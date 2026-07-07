# Extract Elixhauser comorbidity correlation matrices from 4CE .rda files.
#
# Produces: site_elix_correlations.csv — one row per (site, neuro_type, comorbidity_pair)
# with the pairwise correlation between Elixhauser categories.

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else "data/4ce_neuro/results"
out_dir <- if (length(args) > 1) args[2] else "data/4ce_neuro"

rda_files <- list.files(data_dir, pattern = "*_results.rda", full.names = TRUE)
cat(sprintf("Found %d .rda files\n", length(rda_files)))

all_rows <- list()
profile_rows <- list()

for (f in rda_files) {
  env <- new.env()
  tryCatch(load(f, envir = env), error = function(e) NULL)
  obj_name <- ls(env)[1]
  if (is.null(obj_name)) next
  r <- get(obj_name, envir = env)
  site <- r$site
  cat(sprintf("  %s\n", site))

  ca <- r$comorbidities$comorb_adults
  if (is.null(ca)) next

  matrices <- list(
    overall = ca$elix_mat,
    CNS = ca$elix_mat_cns,
    PNS = ca$elix_mat_pns
  )

  for (neuro_type in names(matrices)) {
    m <- matrices[[neuro_type]]
    if (is.null(m)) next
    if (!is.matrix(m)) next

    nms <- rownames(m)
    if (is.null(nms)) next

    for (i in seq_len(nrow(m))) {
      for (j in seq(i + 1, ncol(m))) {
        if (j > ncol(m)) break
        val <- m[i, j]
        if (is.na(val)) next
        all_rows[[length(all_rows) + 1]] <- data.frame(
          site = site,
          neuro_type = neuro_type,
          comorbidity_a = nms[i],
          comorbidity_b = nms[j],
          correlation = val,
          stringsAsFactors = FALSE
        )
      }
    }
  }

  # Also extract comorbidity prevalence profiles
  tc <- r$first_hosp_results$tableone_comorbidity_results$tableone_comorbidity_adults
  if (!is.null(tc) && !is.null(tc$demo_table)) {
    dt <- tc$demo_table
    # Get prevalence columns for each comorbidity group
    prop_cols <- grep("^prop_", names(dt), value = TRUE)
    for (pc in prop_cols) {
      comorb <- sub("^prop_", "", pc)
      # Severity row
      sev_idx <- which(dt$variable == "Severity.Severe")
      if (length(sev_idx) > 0 && !is.na(dt[[pc]][sev_idx])) {
        profile_rows[[length(profile_rows) + 1]] <- data.frame(
          site = site,
          comorbidity = comorb,
          prop_severe = dt[[pc]][sev_idx],
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

if (length(all_rows) > 0) {
  corr_df <- do.call(rbind, all_rows)
  cat(sprintf("\nCorrelations: %d rows (%d sites)\n", nrow(corr_df), length(unique(corr_df$site))))
  write.csv(corr_df, file.path(out_dir, "site_elix_correlations.csv"), row.names = FALSE)
}

if (length(profile_rows) > 0) {
  prof_df <- do.call(rbind, profile_rows)
  cat(sprintf("Profiles: %d rows (%d sites)\n", nrow(prof_df), length(unique(prof_df$site))))
  write.csv(prof_df, file.path(out_dir, "site_elix_profiles.csv"), row.names = FALSE)
}

cat("Done.\n")
