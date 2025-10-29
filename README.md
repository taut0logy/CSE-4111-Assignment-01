# CSE 4111 Assignment on Classification Models

This repository contains three Jupyter notebooks implementing and comparing two popular classification algorithms—**Multi-Layer Perceptron (MLP)** and **Support Vector Machine (SVM)**—on three distinct medical datasets. Each notebook follows a structured machine learning pipeline from data loading to model evaluation and visualization.

## Table of Contents

1. [Introduction](#introduction)
2. [Models Overview](#models-overview)
   - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
   - [Support Vector Machine (SVM)](#support-vector-machine-svm)
3. [Datasets](#datasets)
4. [Procedure](#procedure)

## Introduction

Classification is a fundamental supervised learning task where the goal is to predict discrete class labels for input samples. This assignment explores two powerful classification techniques:

- **Multi-Layer Perceptron (MLP)**: A feedforward artificial neural network that learns complex, non-linear decision boundaries through multiple layers of neurons.
- **Support Vector Machine (SVM)**: A discriminative classifier that finds optimal hyperplanes to separate classes, often using kernel methods to handle non-linear patterns.

By applying these models to three real-world medical datasets, we demonstrate their effectiveness, compare their performance, and analyze their behavior under different conditions (e.g., with and without dimensionality reduction via PCA).

## Models Overview

### Multi-Layer Perceptron (MLP)

**What is MLP?**

A Multi-Layer Perceptron is a class of feedforward artificial neural network consisting of:

- **Input layer**: Receives the feature vectors
- **Hidden layer(s)**: One or more layers of neurons that learn intermediate representations
- **Output layer**: Produces class predictions

Each neuron applies a weighted sum of its inputs followed by a non-linear activation function (e.g., ReLU, sigmoid). The network learns by adjusting weights through **backpropagation** and gradient descent to minimize classification error.

**Why use MLP?**

- Can model complex, non-linear relationships between features and target classes
- Flexible architecture: number of hidden layers and neurons can be tuned
- Effective on large datasets with sufficient training data

**Implementation Details:**

- `MLPClassifier` from `scikit-learn`
- Architecture: Single hidden layer with 100 neurons (configurable)
- Activation: ReLU (default)
- Optimization: Adam or stochastic gradient descent
- Max iterations: 500-1000 (to allow convergence)
- Random state: 42 (for reproducibility)

**Key Hyperparameters:**

- `hidden_layer_sizes`: Number and size of hidden layers
- `max_iter`: Maximum number of training iterations
- `learning_rate_init`: Initial learning rate for weight updates
- `early_stopping`: Stops training when validation score doesn't improve

### Support Vector Machine (SVM)

**What is SVM?**

A Support Vector Machine is a powerful discriminative classifier that works by:

1. Finding the optimal **hyperplane** that maximally separates classes in feature space
2. Maximizing the **margin**—the distance between the hyperplane and the nearest data points (support vectors) from each class
3. Using **kernel functions** to map data into higher-dimensional spaces where non-linear patterns become linearly separable

**Why use SVM?**

- Effective in high-dimensional spaces (many features)
- Memory efficient: only uses support vectors (subset of training data)
- Versatile: different kernel functions (linear, polynomial, RBF) for various data distributions
- Robust to overfitting, especially in high-dimensional spaces

**Implementation Details:**

- `SVC` from `scikit-learn`
- Kernel: Radial Basis Function (RBF) — transforms data into infinite-dimensional space
- Probability estimates: Enabled (`probability=True`) for ROC curve analysis
- Random state: 42 (for reproducibility)

**Key Hyperparameters:**

- `kernel`: Type of kernel function ('linear', 'poly', 'rbf', 'sigmoid')
- `C`: Regularization parameter (controls trade-off between margin size and classification error)
- `gamma`: Kernel coefficient for RBF (controls influence of individual training samples)

**RBF Kernel:**
The Radial Basis Function kernel is defined as:

$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

where $\gamma$ controls the influence radius of support vectors. It's particularly effective for non-linear classification problems.

## Datasets

### 1. **Breast Cancer Wisconsin (Diagnostic)**

**Source:** Built into `scikit-learn` (`load_breast_cancer`)

**Description:**

- Medical diagnostic dataset for breast cancer classification
- Features computed from digitized images of fine needle aspirate (FNA) of breast masses
- **Task:** Binary classification (Malignant vs. Benign)

**Statistics:**

- Samples: 569 patients
- Features: 30 numeric features (e.g., radius, texture, perimeter, area, smoothness, compactness)
- Classes: 2 (Malignant: 212, Benign: 357)
- Missing values: None

**Features include:** Mean, standard error, and "worst" values for 10 real-valued characteristics of cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).

**Notebook:** `breast_cancer.ipynb`

### 2. **Heart Disease**

**Source:** UCI Machine Learning Repository (ID: 45)

**Description:**

- Classic dataset for predicting presence of heart disease
- Features include patient demographics, symptoms, and test results
- **Task:** Binary classification (Heart Disease present vs. absent)

**Statistics:**

- Samples: 303 patients
- Features: 13 numeric/categorical features
- Classes: Originally 5 levels (0-4), converted to binary (0: no disease, 1-4: disease present)
- Missing values: Present in 'ca' (coronary arteries) and 'thal' (thalassemia) columns

**Key Features:**

- Age, sex, chest pain type (cp)
- Resting blood pressure (trestbps)
- Serum cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting ECG results (restecg)
- Maximum heart rate achieved (thalach)
- Exercise-induced angina (exang)
- ST depression induced by exercise (oldpeak)
- Slope of peak exercise ST segment
- Number of major vessels colored by fluoroscopy (ca)
- Thalassemia (thal)

**Notebook:** `heart_disease.ipynb`

### 3. **Diabetes 130-US Hospitals (1999-2008)**

**Source:** UCI Machine Learning Repository (ID: 296)

**Description:**

- Large-scale clinical dataset from 130 US hospitals over 10 years
- Represents patient encounters and hospital outcomes for diabetes patients
- **Task:** Multi-class classification (patient readmission: <30 days, >30 days, or no readmission)

**Statistics:**

- Samples: ~101,766 patient encounters
- Features: 50+ features (demographics, diagnoses, medications, procedures)
- Classes: 3 readmission categories
- Missing values: Present in multiple columns
- Data types: Mix of numeric and categorical features

**Key Features:**

- Patient demographics (age, gender, race)
- Admission details (type, source, time in hospital)
- Medical history (number of procedures, medications, diagnoses)
- Lab results and test indicators
- Medication changes
- Discharge disposition

**Note:** This notebook focuses on numeric features after preprocessing and handles missing values through median imputation.

**Notebook:** `thyroid_diabetes.ipynb`

## Procedure

Each notebook follows a standardized machine learning pipeline consisting of the following steps:

### 1. **Environment Setup**

```python
%pip install matplotlib pandas scikit-learn ucimlrepo seaborn numpy
```

Install required Python libraries for data manipulation, machine learning, and visualization.

### 2. **Data Loading**

**For breast cancer (scikit-learn built-in):**

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
```

**For heart disease and diabetes (UCI repository):**

```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=45)  # or id=296 for diabetes
X = ds.data.features
y = ds.data.targets
```

### 3. **Exploratory Data Analysis (EDA)**

- Print dataset shape: number of samples and features
- Check for missing values in each column
- Examine class distribution (for imbalance detection)
- Display sample rows and basic statistics

### 4. **Data Preprocessing**

**Missing Value Imputation:**

```python
missing_cols = X.columns[X.isnull().any()].tolist()
if missing_cols:
    for col in missing_cols:
        X[col] = X[col].fillna(X[col].median())
```

Replace missing numeric values with column-wise median.

**Feature Selection:**

- For diabetes dataset: select only numeric features for simplicity
- Remove highly correlated or redundant features (optional)

**Target Transformation:**

- Heart disease: convert multi-class target to binary (0 vs. 1-4)

```python
y_binary = (y['num'] > 0).astype(int)
```

### 5. **Train-Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

- **80-20 split:** 80% training, 20% testing
- **Stratification:** Maintains class distribution in both sets
- **Random state:** Ensures reproducibility

### 6. **Pipeline Construction**

**Why pipelines?**
Pipelines ensure that preprocessing steps (like scaling) are applied consistently to training and test data, preventing data leakage.

**MLP Pipeline:**

```python
mlp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))
])
```

**SVM Pipeline:**

```python
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, random_state=42))
])
```

**StandardScaler:**

- Standardizes features by removing mean and scaling to unit variance: $z = \frac{x - \mu}{\sigma}$
- Essential for MLP (faster convergence) and SVM (scale-sensitive algorithms)

### 7. **Cross-Validation Setup**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

```

- **5-fold cross-validation:** Splits training data into 5 folds
- **Stratified:** Each fold maintains the original class distribution
- **Shuffle:** Randomizes data before splitting

### 8. **Model Training and Evaluation**

**Custom Evaluation Function:**

```python
def evaluate_model(pipe, X_tr, y_tr, X_te, y_te, cv):
    # Cross-validated predictions on training data
    y_pred_cv = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict")
    
    # Calculate metrics
    p, r, f1, _ = precision_recall_fscore_support(y_tr, y_pred_cv, average="weighted")
    acc_cv = accuracy_score(y_tr, y_pred_cv)
    
    # Fit on full training set and predict on test set
    pipe.fit(X_tr, y_tr)
    y_pred_test = pipe.predict(X_te)
    
    # Test metrics
    p_t, r_t, f1_t, _ = precision_recall_fscore_support(y_te, y_pred_test, average="weighted")
    acc_t = accuracy_score(y_te, y_pred_test)
    
    return {
        "cv": {"precision": p, "recall": r, "f1": f1, "accuracy": acc_cv},
        "test": {"precision": p_t, "recall": r_t, "f1": f1_t, "accuracy": acc_t},
        "model": pipe,
        "y_pred_test": y_pred_test
    }
```

**Evaluation Metrics:**

- **Precision:** $P = \frac{TP}{TP + FP}$ — Of all positive predictions, how many are correct?
- **Recall (Sensitivity):** $R = \frac{TP}{TP + FN}$ — Of all actual positives, how many did we find?
- **F1-Score:** $F1 = 2 \cdot \frac{P \cdot R}{P + R}$ — Harmonic mean of precision and recall
- **Accuracy:** $Acc = \frac{TP + TN}{Total}$ — Overall correctness

**Why weighted average?**
For multi-class problems (like diabetes), `average='weighted'` computes metrics for each class and averages them weighted by class support (number of samples).

### 9. **Dimensionality Reduction (PCA Experiments)**

**PCA (Principal Component Analysis):**

- Linear transformation that projects data onto orthogonal axes of maximum variance
- Reduces feature space dimensionality while retaining most information

**95% Variance PCA:**

```python
pca95 = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))
])
X_train_95 = pca95.fit_transform(X_train)
X_test_95 = pca95.transform(X_test)
```

Keeps enough principal components to explain 95% of total variance.

**2D PCA Visualization:**

```python
pca2d = PCA(n_components=2)
X_pca2d = pca2d.fit_transform(StandardScaler().fit_transform(X))
```

Projects data to 2D for visualization of class separability.

### 10. **Visualization and Analysis**

**Confusion Matrix:**

```python
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

Visualizes true positives, false positives, true negatives, and false negatives.

**ROC Curve (Receiver Operating Characteristic):**

```python
fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
```

- Plots True Positive Rate vs. False Positive Rate at various thresholds
- **AUC (Area Under Curve):** Summary metric (1.0 = perfect classifier, 0.5 = random)

**Metric Comparison Bar Charts:**

```python
metrics_df = pd.DataFrame([
    {"model": "MLP", **res_mlp["test"]},
    {"model": "SVM", **res_svm["test"]}
])
metrics_df.set_index("model").T.plot.bar()
```

Side-by-side comparison of precision, recall, F1, and accuracy.

**2D PCA Scatter Plot:**
Visualizes how well classes separate in the first two principal components.

### 11. **Results Interpretation**

Compare models based on:

- **Accuracy:** Overall performance (use with caution for imbalanced datasets)
- **Precision/Recall trade-off:** Choose based on application needs
- **F1-Score:** Balanced metric when classes are imbalanced
- **ROC-AUC:** Threshold-independent performance measure
- **PCA performance:** How much accuracy is lost with dimensionality reduction?

### Typical Performance Ranges

| Dataset | MLP Accuracy | SVM Accuracy | Best Model |
|---------|--------------|--------------|------------|
| Breast Cancer | 95-98% | 96-98% | Either (very close) |
| Heart Disease | 75-85% | 80-88% | Often SVM |
| Diabetes | 55-65% | 58-68% | Often SVM |

**Note:** Actual results may vary due to:

- Random initialization (despite setting `random_state`)
- Convergence behavior of MLP
- Hyperparameter settings (default values used)

### Key Observations

1. **SVM tends to be more stable:** Less sensitive to initialization, often achieves consistent performance
2. **MLP can be more powerful:** With proper tuning (more layers, neurons, epochs), can model complex patterns
3. **PCA impact varies:** Some datasets lose minimal accuracy with 95% variance, others show noticeable degradation
4. **Class imbalance matters:** Weighted metrics provide better insight than raw accuracy

## Further Improvements

Potential enhancements for this project:

1. **Hyperparameter Tuning:**
   - Use `GridSearchCV` or `RandomizedSearchCV` to find optimal hyperparameters
   - Tune `C` and `gamma` for SVM
   - Tune `hidden_layer_sizes`, `learning_rate_init`, `alpha` for MLP

2. **Additional Models:**
   - Random Forest Classifier
   - Gradient Boosting (XGBoost, LightGBM)
   - Logistic Regression (baseline)

3. **Feature Engineering:**
   - Create interaction terms
   - Polynomial features
   - Domain-specific feature transformations

4. **Advanced Techniques:**
   - Handle class imbalance with SMOTE or class weights
   - Ensemble methods (voting, stacking)
   - Deep learning with TensorFlow/PyTorch

5. **Cross-Dataset Analysis:**
   - Compare how models generalize across different medical domains
   - Meta-learning: learn which model works best for which data characteristics

---

## References

- **Breast Cancer Dataset:** [UCI ML Repository - Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Heart Disease Dataset:** [UCI ML Repository - Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Diabetes Dataset:** [UCI ML Repository - Diabetes 130-US hospitals](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
- **Scikit-learn Documentation:** [https://scikit-learn.org/](https://scikit-learn.org/)
- **MLP Classifier:** [Scikit-learn MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- **SVM:** [Scikit-learn SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
