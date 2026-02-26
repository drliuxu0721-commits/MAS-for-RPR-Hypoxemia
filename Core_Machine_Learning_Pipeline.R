# ==============================================================================
# 
# MULTIVARIABLE PREDICTION MODELING PIPELINE
# 
# Description: This script contains the core machine learning pipeline, including
#              wrapper-based feature selection, 10x10 repeated cross-validation, 
#              and code for generating calibration and DCA plots with 95% CIs.
# 
# Data Availability: Due to patient privacy regulations, the original patient-level 
#                    dataset cannot be shared. Users can run this script by 
#                    providing their own dataset formatted as described in the comments.
# ==============================================================================

# ==============================================================================
# PART 1: WRAPPER-BASED FEATURE SELECTION
# ==============================================================================
library(mlr3)
library(mlr3verse)
library(mlr3pipelines) 
library(mlr3fselect)
library(future)
library(future.apply)
library(data.table)
library(mlr3learners)      
library(mlr3extralearners) 
library(e1071)
library(coin)
library(earth)
library(randomForestSRC)
library(xgboost)
library(lightgbm)

set.seed(1234)
plan(multicore, workers = 120) 

# ------------------------------------------------------------------------------
# [USER INSTRUCTION]: Load your own dataset here.
# The dataset should be a CSV file where rows represent patients and columns 
# represent features. Ensure there is a binary target column (e.g., "Hypoxemia") 
# coded as "0" (Negative) and "1" (Positive).
# ------------------------------------------------------------------------------
rt <- read.csv("training_cohort.csv", header = TRUE)
task <- as_task_classif(rt, target = 'Hypoxemia', positive = "1")
task$col_roles$stratum <- "Hypoxemia" 

# Pipeline with embedded imputation and class balancing to prevent data leakage
base_pipeline = po("imputesample") %>>%               
  po("imputehist") %>>%                 
  po("fixfactors") %>>%                 
  po("removeconstants") %>>% 
  po("encode", method = "treatment") %>>% 
  po("classbalancing", reference = "major", adjust = "minor")         

learner_names <- c(
  "classif.rpart", "classif.featureless", "classif.gausspr", "classif.kknn",             
  "classif.log_reg", "classif.naive_bayes", "classif.nnet", "classif.ranger",            
  "classif.bart", "classif.earth", "classif.gam", "classif.gamboost",         
  "classif.imbalanced_rfsrc", "classif.gbm", "classif.ctree", "classif.cforest",
  "classif.xgboost", "classif.lightgbm", "classif.glmnet"
)

results_list <- list()

for (lname in learner_names) {
  if (lname %in% c("classif.xgboost", "classif.lightgbm", "classif.gbm")) {
    Sys.setenv(OMP_THREAD_LIMIT = 1, OPENBLAS_NUM_THREADS = 1, OMP_NUM_THREADS = 1)
  }
  
  lrn_base <- tryCatch({
    lrn(lname, predict_type = "prob")
  }, error = function(e) return(NULL))
  
  if (is.null(lrn_base)) next 
  
  glrn <- GraphLearner$new(base_pipeline %>>% lrn_base)
  glrn$predict_type <- "prob"
  
  # AUPRC as the primary optimization metric
  at <- AutoFSelector$new(
    learner = glrn,
    resampling = rsmp("cv", folds = 10), 
    measure = msr("classif.prauc"),            
    terminator = trm("none"),                  
    fselector = fs("sequential")                
  )
  
  res <- tryCatch({
    at$train(task)
    list(Algorithm = lname, 
         Num_Features = length(at$fselect_result$features[[1]]),
         Features = paste(at$fselect_result$features[[1]], collapse = " | "), 
         Internal_PRAUC = at$fselect_result$classif.prauc)
  }, error = function(e) return(NULL))
  
  if (!is.null(res)) {
    results_list[[lname]] <- res
    checkpoint_df <- rbindlist(results_list)
    write.csv(checkpoint_df, "Feature_Selection_Results.csv", row.names = FALSE)
  }
}

# ==============================================================================
# PART 2: MODEL TRAINING AND 10x10 REPEATED CROSS-VALIDATION
# ==============================================================================
# ------------------------------------------------------------------------------
# [USER INSTRUCTION]: The following file is the output generated from PART 1.
# ------------------------------------------------------------------------------
results_df <- read.csv("Feature_Selection_Results.csv", stringsAsFactors = FALSE)
all_learners <- results_df$Algorithm
glrn_list <- list()

for (lname in all_learners) {
  feats_str <- results_df$Features[results_df$Algorithm == lname]
  selected_feats <- unlist(strsplit(feats_str, " \\| "))
  
  graph <- po("select", selector = selector_name(selected_feats), id = paste0("select_", lname)) %>>% 
    po("imputesample") %>>% po("imputehist") %>>% po("fixfactors") %>>% 
    po("imputemode") %>>% po("removeconstants") %>>% 
    po("encode", method = "treatment") %>>% 
    po("classbalancing", reference = "major", adjust = "minor") %>>% 
    lrn(lname, predict_type = "prob")
  
  glrn <- GraphLearner$new(graph)
  glrn$id <- toupper(gsub("classif.", "", lname)) 
  glrn_list[[length(glrn_list) + 1]] <- glrn
}

resampling <- rsmp("repeated_cv", folds = 10, repeats = 10)
design <- benchmark_grid(tasks = task, learners = glrn_list, resamplings = resampling)
bmr <- benchmark(design)

scores <- bmr$score(msrs(c("classif.prauc")))
summary_stats <- scores[, .(
  Mean_PRAUC = mean(classif.prauc, na.rm=TRUE),
  Lower_CI = quantile(classif.prauc, 0.025, na.rm=TRUE),
  Upper_CI = quantile(classif.prauc, 0.975, na.rm=TRUE)
), by = "learner_id"]

summary_stats <- summary_stats[order(-Mean_PRAUC)]
write.csv(summary_stats, "10x10_CV_PRAUC_with_CI.csv", row.names = FALSE)

# ==============================================================================
# PART 3: CALIBRATION CURVES WITH 95% CONFIDENCE INTERVALS
# ==============================================================================
library(ggplot2)
library(dplyr)

plot_calibration <- function(data_file, title_name, line_color, fill_color, out_filename) {
  # ----------------------------------------------------------------------------
  # [USER INSTRUCTION]: The input CSV must contain at least two columns: 
  # 1. 'Hypoxemia': The actual observed outcomes (0 or 1).
  # 2. 'prob': The predicted probabilities generated by the model (0 to 1).
  # ----------------------------------------------------------------------------
  df <- read.csv(data_file, header = TRUE, check.names = FALSE)
  df$obs <- as.numeric(as.character(df$Hypoxemia))
  df$prob <- as.numeric(df$prob)
  
  binned_df <- df %>%
    mutate(bin = ntile(prob, 5)) %>%
    group_by(bin) %>%
    summarise(
      N_patients = n(), 
      mean_pred = mean(prob, na.rm = TRUE),
      mean_obs = mean(obs, na.rm = TRUE),
      se_obs = sqrt(mean_obs * (1 - mean_obs) / n()),
      lower_obs = pmax(0, mean_obs - 1.96 * se_obs),
      upper_obs = pmin(1, mean_obs + 1.96 * se_obs)
    )
  
  p <- ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "darkgray", size = 0.8) +
    geom_smooth(data = df, aes(x = prob, y = obs), method = "loess", se = TRUE, span = 0.8, color = line_color, fill = fill_color, alpha = 0.4, size = 1) +
    geom_errorbar(data = binned_df, aes(x = mean_pred, ymin = lower_obs, ymax = upper_obs), width = 0.02, color = line_color, size = 0.6) +
    geom_point(data = binned_df, aes(x = mean_pred, y = mean_obs), shape = 21, fill = "white", color = line_color, size = 3, stroke = 1.2) +
    theme_minimal(base_size = 14) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = title_name, x = "Predicted Probability", y = "Observed Proportion") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  pdf(file = out_filename, width = 5, height = 5)
  print(p)
  dev.off()
}

plot_calibration("train_prob.csv", "Training Cohort", "#4682B4", "lightblue", "Calibration_Training.pdf")
plot_calibration("test_prob.csv", "Validation Cohort", "#FF4444", "#FF8888", "Calibration_Validation.pdf")

# ==============================================================================
# PART 4: DECISION CURVE ANALYSIS (DCA) WITH BOOTSTRAPPED 95% CIs
# ==============================================================================
library(dcurves)

get_dca_ci <- function(data, prob_col, obs_col, thresholds, n_boot = 500) {
  n <- nrow(data)
  boot_nb <- matrix(NA, nrow = n_boot, ncol = length(thresholds))
  obs_vector <- as.numeric(as.character(data[[obs_col]]))
  prob_vector <- data[[prob_col]]
  
  set.seed(1234)
  for(i in 1:n_boot) {
    idx <- sample(1:n, replace = TRUE)
    obs_b <- obs_vector[idx]
    prob_b <- prob_vector[idx]
    
    nb <- sapply(thresholds, function(pt) {
      tp <- sum(prob_b >= pt & obs_b == 1)
      fp <- sum(prob_b >= pt & obs_b == 0)
      weight <- ifelse(pt == 1, 0, pt / (1 - pt)) 
      (tp / n) - (fp / n) * weight
    })
    boot_nb[i, ] <- nb
  }
  
  ci_lower <- apply(boot_nb, 2, quantile, probs = 0.025, na.rm = TRUE)
  ci_upper <- apply(boot_nb, 2, quantile, probs = 0.975, na.rm = TRUE)
  return(data.frame(threshold = thresholds, lower = ci_lower, upper = ci_upper))
}

generate_beautiful_dca <- function(data_file, title_text, out_file) {
  # ----------------------------------------------------------------------------
  # [USER INSTRUCTION]: Similar to Calibration, ensure the CSV contains 'Hypoxemia' 
  # and 'prob' columns.
  # ----------------------------------------------------------------------------
  data <- read.csv(data_file, header = TRUE)
  thresholds_seq <- seq(0, 0.99, by = 0.01)
  
  ci_data <- get_dca_ci(data, "prob", "Hypoxemia", thresholds_seq, n_boot = 500)
  
  base_plot <- dca(Hypoxemia ~ prob, data = data, thresholds = thresholds_seq, label = list(prob = "MAS")) %>%
    plot(smooth = TRUE)
  
  final_plot <- base_plot +
    geom_ribbon(data = ci_data, aes(x = threshold, ymin = lower, ymax = upper),
                inherit.aes = FALSE, fill = "#4682B4", alpha = 0.15) + 
    ggtitle(label = title_text) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  pdf(out_file, height = 4, width = 5)
  print(final_plot)
  dev.off()
}

generate_beautiful_dca('train_prob.csv', 'Training Cohort', 'DCA_Training.pdf')
generate_beautiful_dca('test_prob.csv', 'Validation Cohort', 'DCA_Validation.pdf')