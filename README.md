# Chemical Reactor Yield Prediction using Machine Learning

> **Physics-Informed Machine Learning for Predicting Desired Product Yield in a Series Reaction System**

## Overview

This project develops a machine learning framework for predicting the yield of the desired product **B** in a chemical reactor governed by the competing reaction pathway:

**A → B → C**

The objective is to learn the nonlinear relationship between reactor operating conditions and the final yield of Product B.

The project combines **chemical-process intuition, physics-informed feature engineering, machine learning model comparison, cross-validation, and competition-style prediction generation**.

---

## Problem Statement

In a series reaction system, the desired intermediate product **B** can continue reacting to form **C**. Therefore, reactor operating conditions must be considered carefully because temperature, flow conditions, reactor length, and reactant concentration can influence the final yield of B.

The goal of this project is to predict:

```text
overall_yield
```

using measurable reactor operating conditions.

---

## Dataset

The training dataset contains **150 observations** and 6 numerical variables.

| Feature                | Description                                  |
| ---------------------- | -------------------------------------------- |
| `flow_rate_L_min`      | Volumetric flow rate of the reactant mixture |
| `concentration_mol_L`  | Inlet concentration of Reactant A            |
| `inlet_temperature_K`  | Feed temperature entering the reactor        |
| `length_m`             | Reactor length                               |
| `jacket_temperature_K` | External heating-jacket temperature          |
| `overall_yield`        | Target yield of Product B                    |

The organizer-provided test dataset contains **50 observations**, with the target withheld for competition evaluation.

---

## Physics-Informed Feature Engineering

Instead of relying only on the raw variables, the project introduces features motivated by reactor behavior.

### 1. Residence Index

```text
Residence_Index = length_m / flow_rate_L_min
```

This provides a relative proxy for the amount of reaction path/residence opportunity.

### 2. Temperature Difference

```text
Temp_Difference = jacket_temperature_K - inlet_temperature_K
```

This represents the thermal difference between the reactor jacket and the incoming feed.

### 3. Heat Exposure

```text
Heat_Exposure = Temp_Difference × Residence_Index
```

This combines thermal difference with the residence-related proxy.

These engineered variables translate basic reactor intuition into compact numerical features for the ML models.

---

## Exploratory Data Analysis

The project includes:

* Target-distribution analysis
* Descriptive statistics
* Feature-target correlation analysis
* Feature importance analysis
* Missing-value checks
* Duplicate-row checks
* Analysis of engineered physics-informed features

The target variable showed substantial variation across the training observations, supporting the use of nonlinear regression models.

---

## Machine Learning Models

The following approaches were evaluated:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

The initial held-out evaluation showed that nonlinear tree-based models performed substantially better than the linear baseline.

### Initial Evaluation

| Model             |    RMSE |     MAE |     R² |
| ----------------- | ------: | ------: | -----: |
| Linear Regression | 31.4408 | 26.9214 | 0.4378 |
| Decision Tree     | 30.6287 | 18.1755 | 0.4665 |
| Random Forest     | 20.0320 | 13.1711 | 0.7718 |

Gradient Boosting and XGBoost were subsequently evaluated using 5-fold cross-validation because of the relatively small labeled dataset.

---

## Cross-Validation Results

| Model             |   Mean RMSE |  Std. RMSE |
| ----------------- | ----------: | ---------: |
| Random Forest     |     20.7917 |     2.9715 |
| Gradient Boosting |     21.7708 |     2.6488 |
| **XGBoost**       | **20.1318** | **1.9719** |

### Model Selection

XGBoost achieved the lowest average cross-validation RMSE and the lowest fold-to-fold standard deviation among the compared nonlinear models.

Therefore, **XGBoost was selected as the final model**.

---

## Final Model

The selected model configuration was:

```python
XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
```

The final model achieved a **5-fold cross-validation RMSE of approximately 20.13**.

---

## Hyperparameter Tuning

A `GridSearchCV` experiment was also performed over:

* `n_estimators`
* `max_depth`
* `learning_rate`
* `subsample`
* `colsample_bytree`

The best configuration found by the search produced a CV RMSE of **22.4175**, which was worse than the earlier default XGBoost configuration.

Therefore, the project retained the simpler XGBoost configuration rather than assuming that additional tuning necessarily improves generalization.

This experiment demonstrates an important ML principle:

> **Model complexity and hyperparameter tuning should be justified by validation performance rather than used simply because they are more sophisticated.**

---

## Feature Importance

The feature-importance analysis highlighted the importance of reactor thermal conditions and residence-related behavior.

Among the leading features were:

* `inlet_temperature_K`
* `jacket_temperature_K`
* `Residence_Index`

The results support the engineering interpretation that **thermal conditions and reaction residence behavior are important predictive signals for Product B yield**.

Feature importance represents predictive association and should not be interpreted as direct causal evidence.

---

## Competition Prediction

The final model was applied to the organizer-provided test dataset.

The prediction pipeline:

```text
Training Data
      ↓
Data Validation
      ↓
Physics-Informed Feature Engineering
      ↓
Model Comparison
      ↓
5-Fold Cross-Validation
      ↓
XGBoost Selection
      ↓
Final Model Training
      ↓
Test Data Feature Engineering
      ↓
Yield Prediction
```

The final submission contains:

```text
50 predictions
1 column
overall_yield
```

The submission was verified to match the required competition format.

---

## Key Takeaways

* Chemical-process intuition can be incorporated into ML through physics-informed features.
* Nonlinear tree-based models were substantially more effective than the linear baseline for this dataset.
* XGBoost achieved the best observed cross-validation RMSE.
* Hyperparameter tuning did not automatically improve performance.
* Cross-validation is particularly important when the labeled dataset is small.
* ML feature importance can provide useful engineering insights, while remaining distinct from causal analysis.

---

## Limitations

* The labeled dataset contains only **150 observations**, so validation estimates have uncertainty.
* The competition test labels are hidden, so the final hidden-test RMSE cannot be determined locally.
* Physics-informed features are proxies for reactor behavior and are not substitutes for a first-principles kinetic model.
* Cross-validation provides an estimate of generalization performance, not a guarantee of leaderboard performance.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Jupyter Notebook / Google Colab

---

## Project Structure

```text
Chemical-Reactor-Yield-Prediction/
│
├── data/
├── notebooks/
│   └── Log_X.ipynb
├── images/
├── submission/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Reproducibility

To reproduce the project:

```bash
git clone <your-repository-url>
cd Chemical-Reactor-Yield-Prediction
pip install -r requirements.txt
```

Then open:

```text
notebooks/Log_X.ipynb
```

and execute the notebook sequentially.

---

## Author

**Tushar Kumar Mishra**

B.S. in Artificial Intelligence & Data Science
Indian Institute of Technology Jodhpur

---

## Acknowledgement

This project was developed as part of a machine-learning competition involving chemical reactor yield prediction and is intended for educational and research-oriented exploration of machine learning applied to chemical-process data.
