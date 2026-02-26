# MAS-for-RPR-Hypoxemia

This repository contains the core machine learning pipeline and supplementary code for our manuscript: 
**"Development and external validation of a machine learning model for preoperative prediction of intractable hypoxemia in repeat lung surgery"**.

## Overview
Intractable hypoxemia is a severe complication during one-lung ventilation (OLV) for repeat pulmonary resection (RPR). This repository provides the R script used to develop and validate the Multivariate Adaptive Splines (MAS) predictive model. 

To ensure full methodological transparency while complying with patient privacy regulations, we have provided the heavily annotated core pipeline. The pipeline demonstrates:
1. **Wrapper-based Feature Selection:** AUPRC-driven sequential selection using 10-fold nested cross-validation.
2. **Data Leakage Prevention:** Integration of imputation and class-balancing (oversampling) strictly within the resampling iterations via the `mlr3pipelines` framework.
3. **Internal Validation:** 10×10 repeated cross-validation for rigorous performance estimation.
4. **Uncertainty Quantification:** Calculation of 95% confidence intervals for Calibration curves and Decision Curve Analysis (DCA) plots using bootstrapping.

## Files Description
* `Core_Machine_Learning_Pipeline.R`: The primary executable R script containing the full modeling workflow and visualization code.

## Usage
To test the pipeline locally:
1. Clone or download this repository.
2. Ensure you have R installed along with the required packages (`mlr3`, `mlr3verse`, `mlr3pipelines`, `ggplot2`, `dcurves`, etc.).
3. Run the sections sequentially in RStudio.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
