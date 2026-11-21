# Skin Disease Analysis and ML Prediction

## Project Overview 

This project focuses on the analysis and classification of six types of skin disorders using Machine Learning techniques. It uses the UCI Dermatology dataset, which contains clinical and histopathological attributes collected from patients.

The project covers the complete Machine Learning workflow, including data collection, data preprocessing, exploratory data analysis (EDA), statistical analysis, feature engineering, feature selection, model development, hyperparameter optimization, model evaluation, probability calibration, and Explainable AI (XAI).

The primary objective is to develop a multiclass classification model that predicts the category of skin disorder based on clinical and histopathological findings.

**Important:** This project is developed for educational and research purposes. It is not a medical device, does not provide a clinical diagnosis, and must not be used as the sole basis for medical or treatment decisions.

## Project Objectives

This project addresses the following requirements:

1. **Data Analysis:** Perform comprehensive exploratory data analysis and statistical analysis of the dermatology dataset.
2. **Machine Learning Prediction:** Develop and compare multiple Machine Learning algorithms to classify six types of skin disorders.
3. **Doctor-Oriented Research Suggestions:** Examine important clinical and histopathological findings and present cautious, research-based observations that may support further investigation by qualified medical professionals.

Additional objectives include:

- Compare multiple classification algorithms.
- Perform feature engineering and feature selection.
- Handle class imbalance.
- Optimize model hyperparameters.
- Evaluate model performance using cross-validation.
- Analyze prediction probabilities and calibration.
- Apply SHAP for model explainability.
- Save the trained Machine Learning pipeline and metadata.

## Dataset Description

**Dataset:** UCI Dermatology Dataset

The dataset contains:

- Total patients: 366
- Total columns: 35 (34 predictors and 1 target variable)
- Clinical attributes: 11
- Histopathological attributes: 22
- Age attribute: 1
- Target classes: 6

The clinical and histopathological attributes are primarily represented as ordinal grades ranging from 0 to 3. Family history is represented as a binary attribute, and Age is measured in years.

There are 8 missing Age values represented by the `?` placeholder. These values are converted into missing values and handled using median imputation within the Machine Learning pipelines.

### Disease Classes

The model classifies the following six categories:

| Class | Disease Name |
|---|---|
| 1 | Psoriasis |
| 2 | Seborrheic dermatitis |
| 3 | Lichen planus |
| 4 | Pityriasis rosea |
| 5 | Chronic dermatitis |
| 6 | Pityriasis rubra pilaris |

## Technologies and Libraries Used

### Programming Language
- Python

### Data Analysis and Visualization
- Pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn
- Imbalanced-learn
- XGBoost
- CatBoost

### Hyperparameter Optimization
- RandomizedSearchCV
- Optuna

### Explainable AI
- SHAP

### Model Saving and Serialization
- Joblib
- JSON

### Development Environment
- Google Colab
- Jupyter Notebook

## Project Workflow

The project follows a structured Machine Learning pipeline.

### 1. Data Loading and Initial Inspection

- Load the dermatology dataset.
- Examine the dataset dimensions.
- Inspect column names and data types.
- Understand the target variable and disease categories.
- Review initial records and dataset structure.

### 2. Data Quality Assessment

- Identify missing values.
- Detect missing Age values represented by `?`.
- Check duplicate records.
- Validate feature ranges.
- Verify target class values.
- Examine data types and possible outliers.

### 3. Data Cleaning and Preprocessing

- Convert `?` values into NaN.
- Convert Age into a numeric data type.
- Apply median imputation for missing values.
- Validate feature values and target classes.
- Use Scikit-learn pipelines to perform preprocessing within training folds.

### 4. Exploratory Data Analysis (EDA)

The notebook includes 12 visualizations to understand the dataset.

Major analysis areas include:

- Age distribution.
- Disease class distribution.
- Spearman correlation heatmap.
- Feature-wise box plots.
- Violin plots.
- Pair plots.
- Stacked class distributions.
- Individual feature distributions.
- Missing-value analysis.
- Feature frequency analysis.
- PCA-based multivariate visualization.
- Class-wise feature profiles.

### 5. Statistical Analysis and Hypothesis Testing

Statistical methods are applied to examine relationships between features and disease categories.

Techniques include:

- Kruskal-Wallis H test.
- Chi-square test of independence.
- Cramér's V effect size.
- Epsilon-squared effect size.
- Holm multiple-testing correction.

These analyses examine associations between features and disease classes. Association does not establish causation.

### 6. Clinical vs Histopathological Analysis

Three feature groups are compared:

- Clinical features: 11 predictors.
- Histopathological features: 22 predictors.
- Combined features: all 34 predictors.

Logistic Regression and Random Forest are evaluated using repeated stratified cross-validation.

### 7. Feature Engineering

The project investigates additional features, including:

- Clinical feature totals.
- Histopathological feature totals.
- Histological pattern groups.
- Age groups.
- Feature interaction terms.
- Standardization for scale-sensitive algorithms.

The engineered totals are exploratory summaries and are not validated medical scores.

### 8. Feature Selection

The following techniques are investigated:

- Mutual Information.
- ANOVA F-score.
- Random Forest feature importance.
- Permutation importance.
- Recursive Feature Elimination (RFE).
- Principal Component Analysis (PCA).

Feature selection experiments are evaluated using training data and cross-validation pipelines.

### 9. Model Development

Multiple Machine Learning algorithms are compared.

**Baseline algorithms:**

- Dummy Classifier.
- Logistic Regression.
- Decision Tree.
- Gaussian Naive Bayes.

**Advanced algorithms:**

- K-Nearest Neighbors (KNN).
- Support Vector Machine (SVM).
- Random Forest.
- Extra Trees.
- Gradient Boosting.
- HistGradientBoosting.
- XGBoost.
- CatBoost.

A total of 12 classification algorithms are evaluated against a Dummy Classifier benchmark.

### 10. Class Imbalance Handling

Different strategies are investigated:

- Class weighting.
- Random oversampling.
- SMOTE.
- No-resampling baseline.

Resampling experiments are performed inside training folds. SMOTE-generated values are also examined because the original clinical and histopathological grades are ordinal.

### 11. Hyperparameter Optimization

Two optimization approaches are implemented:

- RandomizedSearchCV.
- Optuna with Tree-structured Parzen Estimator (TPE).

The strongest candidate models are tuned using training data. Independent repeated cross-validation is used to re-estimate performance and reduce optimism from hyperparameter selection.

### 12. Model Evaluation

Model evaluation uses several performance metrics:

- Accuracy.
- Precision.
- Recall.
- Macro-F1 score.
- Weighted F1 score.
- Balanced accuracy.
- Matthews Correlation Coefficient (MCC).
- ROC-AUC.
- Log-loss.
- Brier score.

The primary model-selection metric is macro-F1 because the dataset contains six classes with different sample counts.

### 13. Probability Calibration and Uncertainty Analysis

The project compares:

- Uncalibrated predictions.
- Sigmoid calibration.
- Isotonic calibration.

Additional diagnostics include:

- Multiclass Brier score.
- Per-class Brier scores.
- Expected Calibration Error (ECE).
- Reliability diagrams.
- Probability distribution analysis.
- Exact zero and one probability checks.
- Repeated-fold calibration stability.
- Out-of-fold versus test-set comparison.

The model's probabilities are research-model scores, not validated clinical disease risks. High confidence does not guarantee a correct prediction.

### 14. Explainable AI Using SHAP

SHAP is used to investigate how individual features contribute to the predictions of the selected model.

The analysis includes:

- Global feature importance.
- Class-specific feature contributions.
- Individual patient explanations.
- Comparison with permutation importance.

SHAP explains model behaviour. It does not establish medical causation or diagnostic criteria.

### 15. Misclassification Analysis

The project examines incorrectly classified patients using:

- Actual disease class.
- Predicted disease class.
- Full six-class probability distribution.
- Rank of the actual class.
- Review-flag status.

Pseudonymous case IDs are used in the revised analysis instead of displaying original dataset row numbers or patient feature values.

### 16. Research-Based Doctor Recommendations

The project examines how clinical and histopathological findings contribute to classification.

The recommendations discuss:

- The importance of combining clinical and histopathological evidence.
- Features associated with different disease categories.
- Disease pairs that are difficult for the model to distinguish.
- Prediction uncertainty and specialist review.
- The importance of examining all class probabilities.

These findings are educational and research-oriented. They are not clinical guidance, a triage protocol, or a substitute for qualified medical assessment.

## Model Performance

The notebook's stored evaluation results report the following performance for the selected Extra Trees model with isotonic probability calibration.

### Final Test Set Results

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 97.30% |
| Macro Precision | 97.22% |
| Macro Recall | 97.22% |
| Macro-F1 Score | 96.97% |
| Weighted F1 Score | 97.30% |
| Balanced Accuracy | 97.22% |
| Matthews Correlation Coefficient | 0.9670 |
| ROC-AUC (Macro, OvR) | 99.93% |
| Log-loss | 0.0712 |
| Brier Score | 0.0436 |

### Cross-Validation Results

The selected Extra Trees configuration achieved:

- Repeated cross-validation macro-F1: approximately 98.4%.
- Standard deviation: approximately 0.018.
- Evaluation: 5-fold stratified cross-validation repeated 3 times using the independent repeated-fold evaluation reported in the notebook.

### Selected Model

- Algorithm: Extra Trees Classifier.
- Hyperparameter optimization: RandomizedSearchCV.
- Selected number of estimators: 206.
- Probability calibration: Isotonic.
- Primary selection metric: Macro-F1.

**Evaluation limitations:** The final test set contains 74 patients, including only 4 patients from class 6. The dataset contains only 366 patients and has no external or prospective clinical validation. The reported scores must not be interpreted as evidence of real-world clinical accuracy.

## Model Saving and Artifacts

The notebook saves the trained model pipelines and metadata using Joblib and JSON.

Expected model artifacts generated when the relevant notebook cells are executed:

```text
models/
    best_skin_disorder_model.joblib
    best_skin_disorder_model_full_data.joblib
    model_metadata.json
```

The artifacts represent:

- `best_skin_disorder_model.joblib`: Model fitted on the training split and associated with the reported test evaluation.
- `best_skin_disorder_model_full_data.joblib`: Model refitted using all 366 patients after evaluation.
- `model_metadata.json`: Model configuration, feature order, class mapping, evaluation metrics, review threshold, and software version information.

The two model files have different purposes. The full-data model does not have an independent held-out test evaluation.

## Project Structure

The following is a suggested repository structure for organizing the project:

```text
Skin-Disease-Analysis-and-ML-Prediction/
│
├── notebooks/
│   └── PRCP_1027_Skin_Disorder_Prediction.ipynb
│
├── data/
│   ├── raw/
│   │   └── dataset_35_dermatology.csv
│   └── processed/
│
├── models/
│   ├── best_skin_disorder_model.joblib
│   ├── best_skin_disorder_model_full_data.joblib
│   └── model_metadata.json
│
├── artifacts/
│   ├── figures/
│   └── evaluation/
│
├── app/
│   └── (Future Streamlit application)
│
├── api/
│   └── (Future FastAPI backend)
│
├── tests/
│   └── (Future unit and API tests)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

This structure is a proposed organization. The application, API, tests, Docker configuration, and supporting files should only be considered implemented after they have been created and tested.

## Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Skin-Disease-Analysis-and-ML-Prediction.git
```

### Step 2: Navigate to the Project

```bash
cd Skin-Disease-Analysis-and-ML-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 4: Install Required Libraries

Install the libraries used in the notebook:

```bash
pip install numpy pandas scipy statsmodels matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost optuna shap joblib jupyter
```

For reproducible execution, use the exact package versions recorded by the notebook's saved metadata where available.

### Step 5: Configure the Dataset Path

Place the dataset in the `data/raw/` directory.

Update the `DATA_PATH` variable in the notebook to point to the actual CSV file location.

The uploaded notebook currently contains a Google Colab-specific dataset path, so this adjustment is required when running it locally.

### Step 6: Run the Notebook

```bash
jupyter notebook
```

Open:

```text
PRCP_1027_Skin_Disorder_Prediction.ipynb
```

Run the notebook cells in order to reproduce the analysis, model evaluation, explanations, and saved artifacts.

## Challenges Faced and Solutions

| Challenge | Approach |
|---|---|
| Missing Age values | Converted `?` to NaN and applied median imputation inside pipelines. |
| Class imbalance | Compared class weighting, random oversampling, and SMOTE. |
| Small dataset | Used stratified cross-validation and bootstrap confidence intervals. |
| Similar disease characteristics | Used class-wise metrics, confusion matrices, and error analysis. |
| Ordinal feature values | Investigated rank-based statistical methods and tree-based algorithms. |
| Correlated features | Evaluated feature selection, SHAP, and permutation importance. |
| Different feature scales | Applied standardization inside pipelines for scale-sensitive models. |
| Overfitting risk | Used cross-validation, independent repeated folds, and a held-out test set. |
| Model interpretability | Applied SHAP and permutation importance. |
| Probability uncertainty | Compared calibration approaches and added uncertainty diagnostics. |
| Reproducibility | Used fixed random seeds and saved model pipelines and metadata. |

## Limitations

- The dataset contains only 366 patients.
- The data comes from a single research dataset.
- Class 6 has a very small number of patients.
- Clinical and histopathological grades may be subject to observer variability.
- There is no external or prospective validation.
- Model probabilities are not validated clinical risk estimates.
- SHAP and feature importance do not establish causation.
- A high-confidence prediction can still be incorrect.
- The model must not be used independently for diagnosis or treatment decisions.

## Future Enhancements

Potential future improvements include:

1. External validation using data from additional hospitals.
2. Prospective evaluation with qualified medical professionals.
3. Investigation of inter-observer variability.
4. Improved handling of missing and noisy histopathological findings.
5. Experiment tracking using MLflow.
6. Development of a Streamlit dashboard.
7. Development of a FastAPI prediction service.
8. Unit testing and API testing.
9. Docker-based deployment.
10. Investigation of ordinal-aware Machine Learning algorithms.

These are proposed enhancements and should not be interpreted as completed features unless implemented and tested.

## Conclusion

This project demonstrates an end-to-end Machine Learning workflow for multiclass skin disorder classification using structured clinical and histopathological data.

It includes exploratory data analysis, statistical testing, feature engineering, feature selection, comparison of 12 Machine Learning algorithms, hyperparameter optimization, model evaluation, probability calibration, uncertainty analysis, and Explainable AI using SHAP.

The selected Extra Trees model achieved a test accuracy of 97.30% and macro-F1 score of 96.97% on the 74-patient held-out test set.

Despite the promising results, the small dataset and absence of external clinical validation limit real-world conclusions. Further validation and specialist assessment are necessary before any consideration of clinical use.

## Disclaimer

This project is intended strictly for educational and research purposes. It is not a medical device and does not provide medical diagnoses or treatment recommendations. Predictions must never be used as the sole basis for clinical decisions. All medical assessments must be performed by qualified healthcare professionals.

---

**Project:** PRCP-1027 – Skin Disorder Prediction  
**Domain:** Healthcare / Dermatology  
**Approach:** Multiclass Machine Learning Classification  
**Dataset:** UCI Dermatology Dataset  
**Primary Model:** Extra Trees Classifier with Isotonic Calibration  
**Language:** Python
