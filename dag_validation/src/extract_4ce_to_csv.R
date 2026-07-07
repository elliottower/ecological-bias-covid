# Extract 4CE neurological outcome counts from .rda files to CSV.
# Each row is a site, columns are ICD code counts + total patients.

library(tools)

args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) > 0) args[1] else "data/4ce_neuro/results"
out_file <- if (length(args) > 1) args[2] else "data/4ce_neuro/site_neuro_counts.csv"

rda_files <- list.files(data_dir, pattern = "*_results.rda", full.names = TRUE)
cat(sprintf("Found %d .rda files in %s\n", length(rda_files), data_dir))

icd_codes <- c("F29", "G03", "G04", "G40", "G44", "G45", "G46",
               "G61", "G72", "G93", "H54", "I60", "I61", "I62",
               "I67", "M60", "R27", "R41", "R42", "R43")

icd_labels <- c(
  F29 = "psychiatric", G03 = "meningitis", G04 = "encephalitis",
  G40 = "seizure", G44 = "headache", G45 = "tia",
  G46 = "cerebrovascular_syndrome", G61 = "neuropathy",
  G72 = "myopathy", G93 = "consciousness",
  H54 = "vision", I60 = "subarachnoid_hemorrhage",
  I61 = "intracerebral_hemorrhage", I62 = "other_hemorrhage",
  I67 = "cerebrovascular_disease", M60 = "myositis",
  R27 = "coordination", R41 = "cognition",
  R42 = "dizziness", R43 = "smell_taste"
)

results <- data.frame()

for (f in rda_files) {
  env <- new.env()
  load(f, envir = env)
  obj_name <- ls(env)[1]
  r <- get(obj_name, envir = env)

  site <- r$site
  cat(sprintf("  Processing %s\n", site))

  demo <- r$first_hosp_results$icd_tables$icd_tables_adults$demo_table
  if (is.null(demo)) {
    cat(sprintf("    Skipping %s (no adult ICD table)\n", site))
    next
  }

  # Total patients (NN = no neuro code)
  nn_col <- paste0("n_var_NN")
  total_patients <- NA
  if (nn_col %in% names(demo)) {
    # Row 1 is female count, row 2 is male count for sex variable
    # Sum them for total
    total_patients <- sum(demo[[nn_col]][1:2], na.rm = TRUE)
  }

  row <- data.frame(site = site, total_patients = total_patients, stringsAsFactors = FALSE)

  for (icd in icd_codes) {
    col_name <- paste0("n_var_", icd)
    if (col_name %in% names(demo)) {
      # Sum female + male counts (rows 1-2)
      count <- sum(demo[[col_name]][1:2], na.rm = TRUE)
      row[[paste0("count_", icd)]] <- count
      row[[paste0("rate_", icd)]] <- count / total_patients
    } else {
      row[[paste0("count_", icd)]] <- NA
      row[[paste0("rate_", icd)]] <- NA
    }
  }

  results <- rbind(results, row)
}

cat(sprintf("\nExtracted data for %d sites\n", nrow(results)))
cat(sprintf("Writing to %s\n", out_file))
dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
write.csv(results, out_file, row.names = FALSE)

# Print summary
cat("\nSite summary:\n")
for (i in 1:nrow(results)) {
  n_measured <- sum(!is.na(results[i, grep("^count_", names(results))]))
  cat(sprintf("  %s: %d patients, %d/%d ICD codes measured\n",
              results$site[i], results$total_patients[i], n_measured, length(icd_codes)))
}
