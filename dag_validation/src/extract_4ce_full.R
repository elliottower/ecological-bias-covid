# Extract rich site-level data from 4CE .rda files for causal inference.
#
# Produces two CSVs:
#   1. site_demographics.csv — per-site rates by neuro type (None/Peripheral/Central)
#   2. site_effects.csv — per-site PMI estimates (severity, mortality) with SEs
#
# These feed into CD-NOD for causal discovery and TMLE for effect estimation.

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else "data/4ce_neuro/results"
out_dir <- if (length(args) > 1) args[2] else "data/4ce_neuro"

rda_files <- list.files(data_dir, pattern = "*_results.rda", full.names = TRUE)
cat(sprintf("Found %d .rda files in %s\n", length(rda_files), data_dir))

demo_rows <- list()
effect_rows <- list()
elix_data <- list()

for (f in rda_files) {
  env <- new.env()
  tryCatch({
    load(f, envir = env)
  }, error = function(e) {
    cat(sprintf("  Error loading %s: %s\n", f, e$message))
    return(NULL)
  })

  obj_name <- ls(env)[1]
  if (is.null(obj_name)) next
  r <- get(obj_name, envir = env)

  site <- r$site
  cat(sprintf("  Processing %s\n", site))

  # === Demographics by neuro type ===
  demo <- r$first_hosp_results$tableone_results$tableone_adults
  if (is.null(demo)) next
  dt <- demo$demo_table
  if (is.null(dt)) next

  # Extract counts per neuro type
  neuro_types <- c("None", "Peripheral", "Central")
  for (nt in neuro_types) {
    n_col <- paste0("n_var_", nt)
    p_col <- paste0("prop_", nt)
    if (!(n_col %in% names(dt))) next

    n_total <- sum(dt[[n_col]][dt$variable %in% c("sex.female", "sex.male")], na.rm = TRUE)
    if (n_total == 0) next

    get_prop <- function(var_name) {
      idx <- which(dt$variable == var_name)
      if (length(idx) == 0) return(NA)
      dt[[p_col]][idx]
    }

    get_count <- function(var_name) {
      idx <- which(dt$variable == var_name)
      if (length(idx) == 0) return(NA)
      dt[[n_col]][idx]
    }

    row <- data.frame(
      site = site,
      neuro_type = nt,
      n_patients = n_total,
      prop_female = get_prop("sex.female"),
      prop_age_18to25 = get_prop("age_group.18to25"),
      prop_age_26to49 = get_prop("age_group.26to49"),
      prop_age_50to69 = get_prop("age_group.50to69"),
      prop_age_70to79 = get_prop("age_group.70to79"),
      prop_age_80plus = get_prop("age_group.80plus"),
      prop_white = get_prop("race.white"),
      prop_black = get_prop("race.black"),
      prop_severe = get_prop("Severity.Severe"),
      prop_deceased = get_prop("Survival.Deceased"),
      prop_readmitted = get_prop("readmitted.TRUE"),
      n_severe = get_count("Severity.Severe"),
      n_deceased = get_count("Survival.Deceased"),
      n_readmitted = get_count("readmitted.TRUE"),
      stringsAsFactors = FALSE
    )
    demo_rows[[length(demo_rows) + 1]] <- row
  }

  # === Other table (Elixhauser scores, time to discharge, etc.) ===
  other <- demo$other_obfus_table
  if (!is.null(other)) {
    for (nt in neuro_types) {
      col <- nt
      if (!(col %in% names(other))) next

      # Parse "mean (sd)" format
      parse_mean_sd <- function(name_pattern) {
        idx <- grep(name_pattern, other$name, ignore.case = TRUE)
        if (length(idx) == 0) return(c(NA, NA))
        val <- other[[col]][idx[1]]
        # Format: "5 (10)" or "0.5 (0.3)"
        m <- regmatches(val, regexec("([0-9.-]+)\\s*\\(([0-9.-]+)\\)", val))
        if (length(m[[1]]) == 3) {
          return(c(as.numeric(m[[1]][2]), as.numeric(m[[1]][3])))
        }
        return(c(NA, NA))
      }

      elix <- parse_mean_sd("Mean Elixhauser")
      discharge <- parse_mean_sd("Mean time to first discharge")
      readm_n <- parse_mean_sd("Mean number of readmissions")

      row <- data.frame(
        site = site,
        neuro_type = nt,
        mean_elixhauser = elix[1],
        sd_elixhauser = elix[2],
        mean_time_discharge = discharge[1],
        sd_time_discharge = discharge[2],
        mean_n_readmissions = readm_n[1],
        sd_n_readmissions = readm_n[2],
        stringsAsFactors = FALSE
      )
      effect_rows[[length(effect_rows) + 1]] <- row
    }
  }

  # === Survival regression PMIs ===
  surv <- r$first_hosp_results$survival_results
  if (!is.null(surv)) {
    timepoints <- c(30, 60, 90)
    for (tp in timepoints) {
      key <- paste0("surv_results_adults_lpca_", tp)
      if (!(key %in% names(surv))) next
      s <- surv[[key]]

      for (outcome in c("severe_reg_elix", "deceased_reg_elix")) {
        if (!(outcome %in% names(s))) next
        reg <- s[[outcome]]
        if (is.null(reg$log.pmi) || is.null(reg$se)) next

        outcome_short <- sub("_reg_elix", "", outcome)
        log_pmi <- as.numeric(reg$log.pmi)
        se <- as.numeric(reg$se)
        pmi_names <- names(reg$log.pmi)
        if (is.null(pmi_names)) pmi_names <- paste0("effect_", seq_along(log_pmi))

        for (i in seq_along(log_pmi)) {
          row <- data.frame(
            site = site,
            outcome = outcome_short,
            timepoint = tp,
            effect_name = pmi_names[i],
            log_pmi = log_pmi[i],
            se = se[i],
            pmi = exp(log_pmi[i]),
            lower = exp(log_pmi[i] - 1.96 * se[i]),
            upper = exp(log_pmi[i] + 1.96 * se[i]),
            stringsAsFactors = FALSE
          )
          effect_rows[[length(effect_rows) + 1]] <- row
        }
      }
    }
  }
}

# Write demographics
demo_df <- do.call(rbind, demo_rows[sapply(demo_rows, function(x) ncol(x) == 17)])
cat(sprintf("\nDemographics: %d rows (%d sites)\n", nrow(demo_df), length(unique(demo_df$site))))
write.csv(demo_df, file.path(out_dir, "site_demographics.csv"), row.names = FALSE)

# Write effects (combine clinical + PMI)
# Separate the two types
clinical_rows <- effect_rows[sapply(effect_rows, function(x) "neuro_type" %in% names(x))]
pmi_rows <- effect_rows[sapply(effect_rows, function(x) "log_pmi" %in% names(x))]

if (length(clinical_rows) > 0) {
  clinical_df <- do.call(rbind, clinical_rows)
  cat(sprintf("Clinical: %d rows\n", nrow(clinical_df)))
  write.csv(clinical_df, file.path(out_dir, "site_clinical.csv"), row.names = FALSE)
}

if (length(pmi_rows) > 0) {
  pmi_df <- do.call(rbind, pmi_rows)
  cat(sprintf("PMI effects: %d rows\n", nrow(pmi_df)))
  write.csv(pmi_df, file.path(out_dir, "site_pmi_effects.csv"), row.names = FALSE)
}

cat("\nDone.\n")
