# MillGuard – Predictive Maintenance

## Project Overview

MillGuard is a predictive maintenance project built on a synthetic industrial dataset that simulates machine operating conditions, product characteristics, and failure events.  
The objective is to **predict machine failure events** and **understand which signals contribute most strongly to failure risk**, while accounting for multicollinearity, class imbalance, and interpretability.

This project emphasizes:
- rigorous exploratory data analysis (EDA)
- correlation and multicollinearity awareness
- interpretable modeling
- failure-focused evaluation (not accuracy-centric)

---

## Dataset Overview

The dataset consists of simulated production and condition-monitoring variables designed to resemble real-world industrial systems.  
It includes thermal, mechanical, operational, and wear-related signals, alongside a binary failure target.

The data supports:
- binary classification (failure vs non-failure)
- feature correlation analysis
- interpretable linear modeling
- robustness testing under multicollinearity

---

## Dataset Features

| Feature | Description |
|------|------|
| **UID** | Unique identifier (1–10,000). Used only as a row index and excluded from modeling. |
| **Product ID** | Composite identifier encoding product quality and serial number. Excluded from modeling. |
| **Type** | Product quality class (`L`, `M`, `H`), representing low, medium, and high quality variants. |
| **Air Temperature [K]** | Ambient air temperature simulated via random walk (μ = 300 K, σ = 2 K). |
| **Process Temperature [K]** | Internal process temperature generated from air temperature + 10 K offset with additional random walk noise (σ = 1 K). |
| **Rotational Speed [rpm]** | Rotational speed derived from constant power input (2860 W) with Gaussian noise. |
| **Torque [Nm]** | Normally distributed torque centered at 40 Nm (σ = 10 Nm), clipped at zero. |
| **Tool Wear [min]** | Cumulative tool wear increasing by product quality: +2 (L), +3 (M), +5 (H). |
| **Machine Failure** | Binary target (1 = failure, 0 = no failure), driven by multiple independent failure mechanisms. |

---

## Exploratory Data Analysis (EDA)

### Correlation Analysis

Correlation analysis reveals several **structurally meaningful relationships**:

- **Air Temperature ↔ Process Temperature (ρ ≈ 0.88)**  
  A strong positive correlation driven by construction of the process temperature signal.  
  This confirms thermal dependency and introduces **intentional multicollinearity**.

- **Rotational Speed ↔ Torque (ρ ≈ -0.88)**  
  A strong inverse relationship consistent with the physical constraint  
  \[
  \text{Power} = \text{Torque} \times \omega
  \]
  This validates the physical realism of the dataset.

- **Machine Failure ↔ Failure Modes (HDF, OSF, PWF)**  
  Machine failure shows the strongest associations with:
  - Heat Dissipation Failure (HDF)
  - Overstrain Failure (OSF)
  - Power Failure (PWF)

  Among these, **HDF exhibits the highest correlation**, indicating that thermal stress is a dominant contributor in the simulated failure logic.

- **RNF (Random Failure)**  
  Shows near-zero correlation with all sensor features, confirming that these events are intentionally unpredictable.

### Implications

- The dataset intentionally mixes:
  - correlated signals (temperature, speed–torque)
  - weakly correlated or independent signals (tool wear, random failures)

- This structure mirrors real industrial data and challenges models to:
  - handle multicollinearity
  - extract independent signal contributions
  - avoid overconfidence from redundant features

---

## Modeling Approach

### Model Choice

A **Logistic Regression pipeline** was selected as the baseline model due to:
- interpretability of coefficients
- robustness with regularization
- suitability for failure probability estimation

The modeling pipeline includes:
- numeric imputation (median)
- feature scaling (StandardScaler)
- categorical encoding (One-Hot Encoding, when applicable)
- class imbalance handling via `class_weight="balanced"`

---

### Train–Test Strategy

- Data split: **80% train / 20% test**
- Stratified sampling applied when both classes are present
- Random state fixed for reproducibility

> ⚠️ Note: This dataset is not time-indexed. In real industrial applications, time-aware splits are strongly recommended.

---

### Hyperparameter Tuning

- Regularization strength (`C`) optimized via **GridSearchCV**
- Evaluation metric: **ROC-AUC**
- 5-fold cross-validation

This balances:
- sensitivity to failure events
- resistance to overfitting under multicollinearity

---

## Model Evaluation

### Performance Metrics

- **ROC-AUC** is used as the primary metric to assess ranking quality under class imbalance.
- **Classification Report** is used to inspect precision, recall, and F1-score.

> Accuracy is intentionally deprioritized, as false negatives (missed failures) are typically the most costly error in predictive maintenance.

---

### Confusion Matrix

The confusion matrix highlights:
- True Failures detected
- Missed Failures (False Negatives)
- False Alarms (False Positives)

This allows evaluation of the operational trade-off between:
- maintenance cost
- failure risk

---

### ROC Curve

The ROC curve visualizes:
- model discrimination capability
- trade-off between false positive rate and true positive rate

A ROC-AUC significantly above 0.5 indicates meaningful predictive signal beyond random guessing.

---

## Feature Importance & Interpretability

Feature importance is derived from **standardized logistic regression coefficients**, allowing direct comparison across features.

Key observations:
- Thermal-related features and failure-mode indicators dominate importance rankings.
- Highly correlated features share importance mass due to regularization.
- Tool wear contributes independent but weaker signal.

> Feature importance reflects **model influence**, not causality.

---

## Key Insights

- Thermal dynamics are the dominant failure drivers in this dataset.
- Multicollinearity is present but manageable with regularization.
- Independent signals (e.g., tool wear) complement correlated features.
- Random failures remain inherently unpredictable, limiting achievable recall.

---

## Limitations

- Dataset is synthetic and does not capture temporal degradation patterns.
- Failure mechanisms are simplified and independent.
- No maintenance action or cost modeling is included.

---

## Future Work

- Time-series modeling with rolling windows
- Cost-sensitive threshold optimization
- Survival analysis for time-to-failure
- Model comparison with tree-based and ensemble methods
- SHAP or permutation-based feature attribution

---

## Disclaimer

This dataset is **synthetic** and intended for educational, benchmarking, and methodological exploration.  
It does not represent any real machine, process, or industrial facility.
