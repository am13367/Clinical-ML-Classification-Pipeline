# Breast Cancer Wisconsin (Diagnostic) Classification
## Final Project Write-Up

**Course:** CS-UY 4563 – Introduction to Machine Learning  
**Team Members:**  
- Saad Iftikhar  
- Ahmed Arkam Mohamed Faisaar  

**Date:** December 2024

---

# A. Introduction

## Dataset Description

**Source:** UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) Dataset  
**Link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

**Dataset Characteristics:**
- **Total Samples:** 569 instances
- **Number of Features:** 30 real-valued features
- **Target Variable:** Binary classification (Malignant vs Benign)
- **Class Distribution:**
  - Benign (Class 0): 357 samples (62.7%)
  - Malignant (Class 1): 212 samples (37.3%)
- **Missing Values:** None
- **Feature Types:** All continuous numerical features

**Feature Description:**

The dataset contains measurements computed from digitized images of fine needle aspirate (FNA) of breast masses. The features describe characteristics of cell nuclei present in the images. For each cell nucleus, 10 properties are measured:

1. Radius (mean of distances from center to perimeter)
2. Texture (standard deviation of gray-scale values)
3. Perimeter
4. Area
5. Smoothness (local variation in radius lengths)
6. Compactness (perimeter² / area - 1.0)
7. Concavity (severity of concave portions of the contour)
8. Concave points (number of concave portions of the contour)
9. Symmetry
10. Fractal dimension ("coastline approximation" - 1)

For each of these 10 properties, three measurements are provided:
- Mean value
- Standard error
- "Worst" value (mean of the three largest values)

This results in 30 features total (10 properties × 3 measurements).

## Machine Learning Task

**Task Type:** Supervised binary classification

**Objective:** Given the 30 numeric features describing cell nuclei properties, predict whether a breast tumor is malignant (cancerous) or benign (non-cancerous).

**Why This Task Is Interesting:**

1. **Clinical Relevance:** Early detection of breast cancer significantly improves patient outcomes, with 5-year survival rates exceeding 90% when diagnosed early.

2. **Medical Decision Support:** Automated classification systems can assist pathologists in making more consistent and accurate diagnoses, reducing inter-observer variability.

3. **Real-World Impact:** Breast cancer affects 1 in 8 women. Accurate classification can:
   - Reduce unnecessary biopsies (false positives)
   - Ensure early treatment for malignant cases (minimize false negatives)
   - Improve prioritization of high-risk patients
   - Reduce healthcare costs while improving patient care

4. **Machine Learning Challenge:** The dataset presents an interesting balance - it's small enough to train quickly but large enough to demonstrate generalization, with complex feature interactions that benefit from careful modeling.

5. **Interpretability:** Unlike many medical AI applications, this task uses well-defined, measurable features that can be explained to medical professionals.

**Dataset Satisfies Requirements:**
- ✓ At least 200 training examples: 341 training samples (60% of 569)
- ✓ At least 10 features: 30 features

---

# B. Exploratory & Unsupervised Analysis

## B.1 Feature Distribution Analysis

We visualized the distributions of key features comparing malignant vs benign tumors. Our analysis revealed:

**Key Observations:**
1. **Malignant tumors consistently show higher values** for most features, particularly:
   - Radius: Malignant tumors have larger cell nuclei
   - Perimeter and Area: Proportionally larger
   - Concavity and Concave Points: More irregular boundaries
   - Texture: Higher standard deviation in gray-scale values

2. **Distribution Shapes:**
   - Most features are right-skewed
   - Malignant class shows higher variance
   - Some features show clear separation between classes
   - Others have significant overlap

3. **Class Separability:**
   - Strong univariate separation for features like "worst concave points"
   - Moderate separation for radius, perimeter, area
   - Weaker separation for symmetry and fractal dimension

## B.2 Correlation Analysis

We computed and visualized a 30×30 correlation matrix to identify relationships between features.

**Key Findings:**

1. **High Correlation Pairs (|r| > 0.9):** We identified 21 feature pairs with very strong correlations:
   - **Radius ↔ Perimeter:** r = 0.998 (nearly perfect)
   - **Radius ↔ Area:** r = 0.987
   - **Radius ↔ Worst Radius:** r = 0.970
   - **Perimeter ↔ Area:** r = 0.987
   - **Worst Radius ↔ Worst Perimeter:** r = 0.994

2. **Implications:**
   - Strong multicollinearity exists, suggesting dimensionality reduction could help
   - Many features are redundant (e.g., radius, perimeter, area measure similar concepts)
   - PCA or feature selection methods could reduce redundancy without losing information

3. **Feature Grouping:**
   - Size features (radius, perimeter, area) form a highly correlated group
   - Shape features (concavity, concave points) form another group
   - Texture-related features show moderate correlations

## B.3 Feature Importance Ranking

Using ANOVA F-test, we ranked features by their individual discriminative power:

**Top 10 Most Important Features:**

| Rank | Feature | F-score |
|------|---------|---------|
| 1 | worst concave points | 964.4 |
| 2 | worst perimeter | 897.9 |
| 3 | mean concave points | 861.7 |
| 4 | worst radius | 860.8 |
| 5 | mean perimeter | 697.2 |
| 6 | worst area | 661.6 |
| 7 | mean radius | 647.0 |
| 8 | mean area | 573.1 |
| 9 | mean concavity | 533.8 |
| 10 | worst concavity | 436.7 |

**Observations:**
- "Worst" measurements (maximum values) are more discriminative than means
- Concave points and shape-related features are highly important
- Size features (radius, perimeter, area) also rank highly
- Texture and symmetry features are less discriminative individually

## B.4 Unsupervised Learning

### Principal Component Analysis (PCA)

**Purpose:** Reduce dimensionality while retaining maximum variance, test if classes separate in lower-dimensional space.

**Results:**
- **Components for 95% variance:** 10 components (reduction from 30)
- **First 2 components:** Explain 63% of total variance
- **First 5 components:** Explain 85% of total variance

**2D PCA Visualization:**
We projected the data onto the first two principal components and colored points by true class. The visualization shows:
- **Good class separation** in the first two PCs
- Benign samples cluster more tightly
- Malignant samples are more spread out
- Some overlap exists, suggesting classification won't be perfect
- Linear separation appears feasible

**Interpretation:**
The strong performance of PCA suggests:
1. Much of the variance can be captured in fewer dimensions
2. The high correlation among original features can be reduced
3. Classes have natural structure in principal component space
4. Dimensionality reduction may improve model generalization

### K-Means Clustering (k=2)

**Purpose:** Test if unsupervised clustering discovers the true class structure.

**Configuration:** k=2 clusters (matching the number of classes)

**Results:**

**Clustering Quality Metrics:**
- **Adjusted Rand Index (ARI):** 0.647
- **Normalized Mutual Information (NMI):** 0.525

**Cluster vs True Label Comparison:**

|            | Cluster 0 | Cluster 1 |
|------------|-----------|-----------|
| **Benign** | 202 | 12 |
| **Malignant** | 21 | 106 |

**Analysis:**
- Cluster 0 primarily captures benign cases (202 out of 214 benign)
- Cluster 1 primarily captures malignant cases (106 out of 127 malignant)
- **Accuracy if clusters = classes:** ~90% ((202+106)/341)
- ARI of 0.647 indicates moderate agreement with true labels

**Interpretation:**
1. **Natural structure exists:** Even without labels, K-Means finds groups that align reasonably with true classes
2. **Classes are separable:** The moderate-to-good clustering performance suggests linear or simple non-linear boundaries may suffice
3. **Some overlap:** 33 samples (21 malignant, 12 benign) are misclassified by unsupervised clustering, indicating:
   - Some tumors have ambiguous feature values
   - Perfect classification may not be achievable
4. **This informed our modeling:** Since unsupervised methods show decent separation, we expect supervised methods to perform well

## B.5 Data Preprocessing

**Missing Values:**
- **Status:** No missing values detected
- **Action:** No imputation required

**Train/Validation/Test Split:**
- **Strategy:** Stratified split to maintain class proportions
- **Ratios:** 60% train / 20% validation / 20% test
- **Sizes:**
  - Training: 341 samples (214 benign, 127 malignant)
  - Validation: 114 samples (71 benign, 43 malignant)
  - Test: 114 samples (72 benign, 42 malignant)

**Feature Scaling:**
- **Method:** StandardScaler (z-score normalization)
- **Parameters:** μ = 0, σ = 1
- **Fit on:** Training set only
- **Transform:** Applied to train, validation, and test sets
- **Rationale:** Required for distance-based algorithms (KNN) and improves convergence for neural networks and logistic regression
- **Note:** Scaling is considered preprocessing, NOT a feature transformation for this project

**No Data Leakage:**
- Scaler fit only on training data
- Validation and test sets transformed using training statistics
- No information from validation/test leaked into training

---

# C. Supervised Modeling

## C.1 Model Families

We trained **three distinct learning models** as required:

### 1. Logistic Regression
- **Type:** Linear classifier with L2 regularization (Ridge)
- **Library:** `sklearn.linear_model.LogisticRegression`
- **Hyperparameter:** C (inverse regularization strength)
- **Range:** C ∈ {0.001, 0.01, 0.1, 1, 10, 100}
- **Interpretation:**
  - Smaller C → stronger regularization → simpler model → higher bias, lower variance
  - Larger C → weaker regularization → more complex model → lower bias, higher variance

### 2. K-Nearest Neighbors (KNN)
- **Type:** Instance-based learning (non-parametric)
- **Library:** `sklearn.neighbors.KNeighborsClassifier`
- **Distance Metric:** Euclidean
- **Weights:** Uniform
- **Hyperparameter:** k (number of neighbors)
- **Range:** k ∈ {1, 3, 5, 7, 9, 15}
- **Interpretation:**
  - Smaller k → more complex decision boundary → low bias, high variance (overfitting risk)
  - Larger k → smoother decision boundary → higher bias, lower variance (underfitting risk)

### 3. Multi-Layer Perceptron (MLP) Neural Network
- **Type:** Feedforward neural network
- **Library:** `sklearn.neural_network.MLPClassifier`
- **Architecture:** Single hidden layer with 16 units
- **Activation:** ReLU
- **Optimizer:** Adam
- **Hyperparameter:** alpha (L2 penalty / regularization strength)
- **Range:** α ∈ {0.0001, 0.001, 0.01, 0.1, 0.5, 1.0}
- **Interpretation:**
  - Smaller alpha → weaker regularization → more complex model → risk of overfitting
  - Larger alpha → stronger regularization → simpler model → risk of underfitting

## C.2 Feature Transformations

Beyond baseline preprocessing (scaling), we applied **three feature transformations** as required:

### Baseline: Raw Scaled Features
- **Description:** StandardScaler normalization only (no additional transformation)
- **Features:** All 30 original features
- **Purpose:** Establish performance baseline

### Z1: PCA Features
- **Method:** Principal Component Analysis
- **Configuration:** 10 components (retains 95% of variance)
- **Transformation:** 30 features → 10 principal components
- **Rationale:**
  - **Reduces multicollinearity:** PCA components are orthogonal by construction
  - **Reduces dimensionality:** Removes redundant information
  - **May reduce overfitting:** Fewer features with most signal retained
  - **Removes noise:** Minor components (discarded) often capture noise
- **When we expect it to help:**
  - KNN: Fewer dimensions reduce curse of dimensionality
  - Logistic Regression: Removes collinearity issues
  - MLP: Faster training, potentially better generalization

### Z2: Polynomial Features (Degree 2)
- **Method:** PolynomialFeatures(degree=2, include_bias=False)
- **Input:** Top 8 features (by F-score)
- **Output:** 44 features (8 original + quadratic terms + interactions)
- **Transformation:** 8 features → 44 features
- **Rationale:**
  - **Captures non-linearity:** Allows linear models to fit non-linear patterns
  - **Feature interactions:** Explicitly models pairwise interactions (e.g., radius × concavity)
  - **Increases model complexity:** Higher capacity to fit training data
- **When we expect it to help:**
  - Logistic Regression: Can now fit non-linear boundaries
  - KNN: May provide better feature combinations for distance calculation
- **Risk:** High dimensionality may cause overfitting without proper regularization

### Z3: SelectKBest Features
- **Method:** Univariate feature selection using ANOVA F-test
- **Configuration:** k=10 (top 10 features by F-statistic)
- **Transformation:** 30 features → 10 features
- **Selected Features:**
  - mean radius, mean perimeter, mean area
  - mean concavity, mean concave points
  - worst radius, worst perimeter, worst area
  - worst concavity, worst concave points
- **Rationale:**
  - **Focus on most discriminative features:** Removes weak/redundant features
  - **Reduces dimensionality:** Similar to PCA but maintains interpretability
  - **Reduces computational cost:** Fewer features to process
  - **May improve generalization:** Removes features that add more noise than signal
- **Difference from PCA:**
  - SelectKBest keeps original features (interpretable)
  - PCA creates new orthogonal features (less interpretable)

## C.3 Total Models Trained

**Calculation:**
- 3 model families
- × 4 feature spaces (raw + 3 transformations)
- × 6 hyperparameter values each
- **= 72 models total**

This satisfies the requirement:
> "3 approaches × (3 transformations + 1 untransformed) × 6 hyperparameter settings = 72 models"

## C.4 Training Process and Rationale

For each of the 72 configurations, we:
1. **Trained on training set** (341 samples)
2. **Evaluated on validation set** (114 samples)
3. **Tracked metrics:** Train accuracy, validation accuracy/precision/recall/F1, overfitting gap

**Reasoning for Choices:**

**Example 1: Logistic Regression with C=0.001**
- **Observation:** Both training and validation accuracy were low (~88%)
- **Diagnosis:** Underfitting (high bias)
- **Action:** Increased C to 0.01, then 0.1
- **Result:** Performance improved, found optimal at C=0.1

**Example 2: KNN with k=1 on Raw features**
- **Observation:** Training accuracy = 100%, validation accuracy = 92%
- **Diagnosis:** Severe overfitting (high variance)
- **Action:** Increased k to 3, 5, 7
- **Result:** k=5 provided best balance (train=97%, val=97%)

**Example 3: MLP on Raw features**
- **Observation:** Both train and val accuracy were very low (~50-60%)
- **Diagnosis:** Convergence issues / underfitting in high dimensions
- **Action:** Applied PCA transformation to reduce dimensions
- **Result:** Performance improved significantly with PCA

**Example 4: Polynomial features with large C**
- **Observation:** Training accuracy very high (99%), validation much lower (92%)
- **Diagnosis:** Overfitting due to high dimensionality (44 features)
- **Action:** Decreased C for stronger regularization
- **Result:** Reduced overfitting gap

**Example 5: SelectKBest features**
- **Observation:** Consistent performance across models
- **Diagnosis:** Focused features provide stable signal
- **Action:** No major issues, worked well across model families
- **Result:** Competitive performance with reduced complexity

---

# D. Table of Results

## D.1 Complete Results Table

A comprehensive CSV file (`detailed_results.csv`) contains all 72 models with the following columns:
- Model (Logistic Regression / KNN / MLP)
- Feature_Space (Raw / PCA / Polynomial / SelectKBest)
- Hyperparameter (C / k / alpha)
- Hyperparam_Value
- Train_Accuracy
- Val_Accuracy
- Val_Precision
- Val_Recall
- Val_F1
- Overfit_Gap (Train_Accuracy - Val_Accuracy)

## D.2 Top 10 Best Models (by Validation F1-Score)

| Rank | Model | Feature Space | Hyperparameter | Val F1 | Val Acc | Val Prec | Val Rec |
|------|-------|---------------|----------------|--------|---------|----------|---------|
| 1 | Logistic Reg | Raw | C=0.1 | **0.9762** | **0.9825** | **1.0000** | 0.9535 |
| 2 | Logistic Reg | SelectKBest | C=1.0 | 0.9647 | 0.9737 | 0.9767 | 0.9535 |
| 3 | Logistic Reg | PCA | C=1.0 | 0.9655 | 0.9737 | 0.9545 | 0.9767 |
| 4 | KNN | Raw | k=5 | 0.9639 | 0.9737 | 0.9535 | 0.9744 |
| 5 | KNN | Raw | k=7 | 0.9639 | 0.9737 | 0.9535 | 0.9744 |
| 6 | KNN | PCA | k=7 | 0.9639 | 0.9737 | 0.9535 | 0.9744 |
| 7 | Logistic Reg | PCA | C=0.1 | 0.9639 | 0.9737 | 1.0000 | 0.9302 |
| 8 | KNN | PCA | k=5 | 0.9635 | 0.9649 | 0.9773 | 0.9512 |
| 9 | Logistic Reg | Raw | C=1.0 | 0.9655 | 0.9737 | 0.9545 | 0.9767 |
| 10 | Logistic Reg | SelectKBest | C=10.0 | 0.9556 | 0.9649 | 0.9348 | 0.9767 |

## D.3 Performance Summary by Model Family

### Logistic Regression
- **Average Val F1:** 0.9199
- **Best Val F1:** 0.9762 (Raw, C=0.1)
- **Best Val Accuracy:** 0.9825 (Raw, C=0.1)
- **Range:** F1 from 0.6129 to 0.9762
- **Observation:** Most consistent performer, especially on raw and PCA features

### K-Nearest Neighbors
- **Average Val F1:** 0.9383
- **Best Val F1:** 0.9639 (Raw, k=5 or k=7)
- **Best Val Accuracy:** 0.9737
- **Range:** F1 from 0.9176 to 0.9639
- **Observation:** Very consistent across all feature spaces, highest average F1

### Multi-Layer Perceptron
- **Average Val F1:** 0.7601
- **Best Val F1:** 0.8571 (Polynomial, α=0.0001)
- **Best Val Accuracy:** 0.8947
- **Range:** F1 from 0.5419 to 0.8571
- **Observation:** Struggled on raw features, performed better with dimensionality reduction

## D.4 Performance Summary by Feature Space

### Raw Features
- **Average Val F1:** 0.8134
- **Best Model:** Logistic Regression (C=0.1), F1=0.9762
- **Observation:** High variance in performance; best single model but also worst (MLP)

### PCA Features
- **Average Val F1:** 0.9004
- **Best Model:** KNN (k=7), F1=0.9639
- **Observation:** Most consistent across models, helped MLP significantly

### Polynomial Features
- **Average Val F1:** 0.8869
- **Best Model:** KNN (k=5), F1=0.9512
- **Observation:** Helped MLP, moderate performance for others

### SelectKBest Features
- **Average Val F1:** 0.8904
- **Best Model:** Logistic Regression (C=1.0), F1=0.9647
- **Observation:** Competitive performance, maintained interpretability

## D.5 Visualizations

The following plots are included in the notebook and output files:

1. **Performance Comparison (07_performance_comparison.png):**
   - 4-panel plot showing validation accuracy, precision, recall, and F1-score
   - Bars grouped by feature space, colored by model family
   - Shows KNN and Logistic Regression dominate performance

2. **Hyperparameter Tuning Curves (08_hyperparameter_tuning.png):**
   - 3 subplots (one per model family)
   - Lines show validation F1 vs hyperparameter index
   - Different colors for different feature spaces
   - Logistic Reg: Peak at C=0.1 on raw features
   - KNN: Stable at k=5-7
   - MLP: Best with very small alpha

3. **Overfitting Analysis (09_overfitting_analysis.png):**
   - Scatter plot: training accuracy vs validation accuracy
   - Points colored by model family
   - Diagonal line shows perfect train=val balance
   - Points above line indicate overfitting
   - KNN shows some overfitting, Logistic Reg very close to diagonal

4. **Feature Space Impact (10_feature_space_impact.png):**
   - Bar plot showing average validation F1 per feature space
   - Error bars show standard deviation
   - Stars indicate best model per feature space
   - PCA shows lowest variance, most reliable performance

**Key Insights from Visualizations:**
- **Logistic Regression on Raw features** achieved highest validation metrics
- **KNN** was most consistent across all feature spaces
- **MLP** struggled significantly, especially on raw features
- **PCA transformation** improved consistency across models
- **Polynomial features** showed high variance - helped some, hurt others

---

# E. Conclusion and Analytical Discussion

## E.1 Best Model Identification

### Winner: Logistic Regression with C=0.1 on Raw Scaled Features

**Configuration:**
- **Algorithm:** Logistic Regression with L2 regularization
- **Feature Space:** Raw features (StandardScaler normalization only)
- **Hyperparameter:** C = 0.1 (moderate regularization)
- **Regularization strength:** λ = 1/C = 10

**Validation Performance:**
- **Accuracy:** 98.25%
- **Precision:** 100.00%
- **Recall:** 95.35%
- **F1-Score:** 97.62%
- **Overfitting Gap:** 0.003 (train acc: 97.95%, val acc: 98.25%)

**Test Performance (Final Evaluation):**
- **Accuracy:** 98.25%
- **Precision:** 100.00%
- **Recall:** 95.24%
- **F1-Score:** 97.56%
- **Confusion Matrix:**
  - True Negatives (Benign correctly identified): 72
  - False Positives (Benign misclassified as Malignant): 0
  - False Negatives (Malignant misclassified as Benign): 2
  - True Positives (Malignant correctly identified): 40

**Why This Model Won:**

1. **Perfect Precision (100%):**
   - Zero false positives - no benign tumors misclassified as malignant
   - Clinical significance: Avoids unnecessary biopsies and patient anxiety

2. **High Recall (95.24%):**
   - Only 2 out of 42 malignant cases missed
   - Excellent detection rate for cancer

3. **Excellent Generalization:**
   - Validation-Test gap: only 0.06% (97.62% → 97.56%)
   - Nearly identical performance on unseen data
   - Model is stable and reliable

4. **Simplicity:**
   - Linear model with only 30 parameters
   - Interpretable: Can explain which features contribute to predictions
   - Fast inference: Suitable for real-world deployment

5. **No Feature Engineering Needed:**
   - Raw features performed best
   - Suggests original feature engineering by domain experts was excellent
   - Polynomial features didn't add value

## E.2 Feature Transformation Impact

### Z1: PCA Features

**When it helped:**
- **KNN:** Reduced curse of dimensionality, improved consistency
- **MLP:** Massive improvement over raw features (F1: 0.54 → 0.80)
- **Logistic Regression:** Minor improvement in some configurations

**When it didn't help:**
- **Best single model:** Raw features still outperformed (0.9762 vs 0.9655)

**Why:**
- **Removes multicollinearity:** Beneficial for numerical stability
- **Reduces dimensions:** Helps models struggling with 30 features
- **Information loss:** Some discriminative detail lost in dimensionality reduction
- **Loses interpretability:** Principal components are linear combinations

**Overall Assessment:** ★★★★☆
- Improved average performance across models
- Essential for MLP to work at all
- Didn't achieve the absolute best performance

### Z2: Polynomial Features (Degree 2)

**When it helped:**
- **MLP:** Best MLP performance (F1: 0.8571)
- **Some KNN configurations:** Moderate improvement

**When it didn't help:**
- **Logistic Regression:** Variable performance, often worse than raw
- **Risk of overfitting:** High dimensionality (44 features) without strong regularization

**Why:**
- **Captures non-linearity:** Allows linear models to fit curves
- **Feature interactions:** Models pairwise relationships (e.g., radius × concavity)
- **Curse of dimensionality:** 44 features can overfit without enough regularization
- **Classes already linearly separable:** Non-linearity not needed

**Overall Assessment:** ★★★☆☆
- Useful for specific scenarios (MLP)
- High variance - helped some, hurt others
- Requires careful regularization

### Z3: SelectKBest Features

**When it helped:**
- **Logistic Regression:** Second-best configuration (F1: 0.9647)
- **Competitive across models:** Consistent performance

**When it didn't help:**
- **KNN:** Slightly worse than raw/PCA
- **Worst for KNN:** Among the worst KNN performance

**Why:**
- **Focuses on discriminative features:** Removes noise
- **Maintains interpretability:** Original features retained
- **Ignores feature interactions:** Univariate selection misses dependencies
- **Similar to PCA benefits:** Dimensionality reduction

**Overall Assessment:** ★★★★☆
- Good balance of performance and interpretability
- Reduced computational cost
- Competitive with PCA but interpretable

### Raw Features (Baseline)

**Performance:**
- **Best single model:** Logistic Regression (F1: 0.9762)
- **Variable results:** High variance across models
- **MLP struggled:** Very poor performance (F1: 0.5419)

**Why:**
- **Complete information:** No information loss
- **Domain expertise:** Features engineered by medical imaging experts
- **Multicollinearity present:** Can hurt some models
- **High dimensionality:** 30 features challenging for some algorithms

**Overall Assessment:** ★★★★★ (for Logistic Regression)
- Produced the absolute best model
- But high variance across model families

## E.3 Overfitting and Underfitting Analysis

### Models That Overfit

**1. KNN with k=1 on SelectKBest features:**
- **Train Accuracy:** 100.0%
- **Val Accuracy:** 93.5%
- **Gap:** 7.0%
- **Diagnosis:** **High variance** - memorizing training data
- **Why:** k=1 uses only nearest neighbor, captures noise
- **Fix:** Increased k to 5-7 → gap reduced to ~2%

**2. KNN with k=1 on Polynomial features:**
- **Gap:** 6.1%
- **Diagnosis:** High-dimensional space (44 features) + k=1 → severe overfitting
- **Fix:** Increased k and/or reduced dimensionality

**3. Logistic Regression with C=100 on Raw features:**
- **Gap:** 4.4%
- **Diagnosis:** Very weak regularization → model too complex
- **Fix:** Decreased C to 0.1-1.0 → gap reduced to <1%

### Models That Underfit

**1. MLP on Raw features (all alpha values):**
- **Train Accuracy:** 37-62%
- **Val Accuracy:** 37-62%
- **Both low:** Indicates **high bias** - model too simple for the task
- **Diagnosis:** 
  - Small network (16 units) struggling with 30 features
  - Convergence issues in high-dimensional space
  - Local minima in optimization
- **Fix:** Applied PCA → reduced to 10 features → MLP could learn (F1: 0.80)

**2. Logistic Regression with C=0.001:**
- **Train Accuracy:** 88%
- **Val Accuracy:** 90%
- **Both moderate:** Underfitting - too much regularization
- **Diagnosis:** Strong regularization (λ=1000) constrains model too much
- **Fix:** Increased C to 0.1 → accuracy jumped to 98%

**3. Polynomial features with C=0.001:**
- **Performance:** Poor across metrics
- **Diagnosis:** High-dimensional space (44 features) with extreme regularization → model can't learn
- **Fix:** Increased C to allow model to use features

### Models with Good Balance

**1. Logistic Regression with C=0.1 on Raw features (WINNER):**
- **Train Accuracy:** 97.95%
- **Val Accuracy:** 98.25%
- **Gap:** -0.30% (val slightly better - good sign!)
- **Why balanced:**
  - Moderate regularization (C=0.1) prevents overfitting
  - Sufficient model complexity to capture patterns
  - Well-tuned for this dataset

**2. KNN with k=5-7 on PCA features:**
- **Gap:** 0.9-1.5%
- **Consistent performance:** Similar train and val accuracy
- **Why balanced:**
  - Moderate k smooths decision boundary
  - PCA reduces dimensionality and noise

## E.4 Bias-Variance Trade-Off Analysis

### Logistic Regression: C Parameter

**C = 0.001 (Strong Regularization):**
- **Effect:** High bias, low variance
- **Result:** Underfitting (train: 88%, val: 90%)
- **Model too simple:** Can't capture decision boundary complexity

**C = 0.1 (Optimal):**
- **Effect:** Balanced bias-variance
- **Result:** Excellent performance (train: 97.9%, val: 98.2%)
- **Sweet spot:** Enough flexibility without overfitting

**C = 100 (Weak Regularization):**
- **Effect:** Low bias, high variance
- **Result:** Overfitting (train: 100%, val: 95.6%)
- **Model too complex:** Fits training noise

### KNN: k Parameter

**k = 1 (Maximum Complexity):**
- **Effect:** Low bias, high variance
- **Result:** Overfitting (train: 100%, val: 92-93%)
- **Decision boundary:** Captures every training point detail

**k = 5-7 (Optimal):**
- **Effect:** Balanced bias-variance
- **Result:** Best performance (train: 97%, val: 97%)
- **Decision boundary:** Smooth but captures main patterns

**k = 15 (Maximum Smoothness):**
- **Effect:** Higher bias, lower variance
- **Result:** Slight underfitting (val: 94-95%)
- **Decision boundary:** Too smooth, misses some detail

### MLP: Alpha Parameter and Architecture

**Alpha = 0.0001 (Weak Regularization):**
- **On PCA:** Good performance (F1: 0.80)
- **On Polynomial:** Best MLP (F1: 0.86)
- **Needs low alpha:** Small network (16 units) benefits from flexibility

**Alpha = 1.0 (Strong Regularization):**
- **Result:** Underfitting across all feature spaces
- **Network too constrained:** Can't learn patterns

**Architecture Choice:**
- **Single layer, 16 units:** Relatively simple
- **Bias-variance:** Higher bias (limited capacity) but lower variance (less overfitting risk)
- **Trade-off:** Couldn't match linear models on this dataset

## E.5 Regularization Effects Summary

| Hyperparameter | Increasing Value | Effect on Bias | Effect on Variance | Optimal Value |
|----------------|------------------|----------------|-------------------|---------------|
| C (LogReg) | Weaker regularization | Decrease ↓ | Increase ↑ | C = 0.1 |
| k (KNN) | More neighbors | Increase ↑ | Decrease ↓ | k = 5-7 |
| α (MLP) | Stronger regularization | Increase ↑ | Decrease ↓ | α = 0.0001 |

## E.6 Key Insights

### 1. Simplicity Won
Despite access to polynomial features and neural networks, a **simple linear model** (Logistic Regression) achieved the best performance. This teaches us:
- **Occam's Razor:** Simplest model that works is often best
- **Linear separability:** Classes can be separated by a hyperplane
- **Feature quality:** Well-engineered features by domain experts reduce need for complex models

### 2. Feature Engineering vs Model Complexity
The original 30 features, designed by medical imaging experts, were so well-crafted that:
- Polynomial features didn't add value
- Simple scaling was sufficient preprocessing
- **Lesson:** Good features > complex models

### 3. Dimensionality Reduction Benefits Vary
- **PCA helped:** Models struggling with 30 features (MLP, some KNN)
- **PCA hurt:** Best single model (LogReg on raw: 97.62% > LogReg on PCA: 96.55%)
- **Lesson:** Apply dimensionality reduction when model complexity or overfitting is an issue

### 4. KNN Was Most Consistent
- **Lowest variance:** Performance stable across feature spaces
- **Always competitive:** Never failed catastrophically
- **Lesson:** KNN is a robust baseline, even if not the absolute best

### 5. Neural Networks Struggled
Our MLP (16 hidden units) couldn't match simpler models:
- **Possible reasons:**
  - Network too small for raw features
  - Not enough data (341 training samples)
  - Classes linearly separable → NN overkill
- **Lesson:** Neural networks need either more data, larger networks, or more complex problems

### 6. Regularization is Critical
Small changes in regularization had huge impacts:
- **Logistic Regression:** C: 0.01→0.1 improved F1 from 0.95 to 0.98
- **KNN:** k: 1→5 reduced overfitting from 7% gap to 2%
- **Lesson:** Always tune regularization, even on simple models

### 7. Perfect Precision is Achievable
Multiple configurations achieved 100% precision:
- **Clinical value:** No false alarms → fewer unnecessary biopsies
- **Trade-off:** Some sacrifice in recall (95% vs 98%)
- **Lesson:** In medical contexts, optimizing for specific metrics matters

## E.7 Limitations and Future Work

### Current Limitations

**1. Small Neural Network:**
- Only 16 hidden units in a single layer
- May not have capacity to capture complex patterns
- **Future:** Try deeper networks (64-32-16 architecture) or wider (128 units)

**2. Limited Hyperparameter Search:**
- Grid search over 6 values per parameter
- May have missed optimal values
- **Future:** Use Bayesian optimization or more fine-grained search

**3. No Ensemble Methods:**
- Didn't try Random Forest, Gradient Boosting, or stacking
- **Future:** Combine predictions from Logistic Regression + KNN

**4. Equal Cost Assumption:**
- Treated all errors equally
- In reality: Missing cancer (false negative) is worse than false alarm
- **Future:** Implement cost-sensitive learning or adjust decision threshold

**5. Single Dataset:**
- Only tested on Wisconsin Diagnostic dataset
- **Future:** Validate on external datasets from different hospitals

**6. No Feature Engineering:**
- Used features as-is
- **Future:** Create domain-specific features (e.g., ratios, differences)

**7. Imbalanced Classes:**
- 62.7% benign, 37.3% malignant
- Didn't use class weights or SMOTE
- **Future:** Try oversampling minority class

### Recommended Next Steps

**Short Term (improve current approach):**
1. **Ensemble Methods:**
   - Voting classifier: Logistic Regression + KNN
   - Stacking: Use multiple models' predictions as meta-features
   - Expected improvement: 0.5-1% F1-score boost

2. **Cost-Sensitive Learning:**
   - Weight false negatives 5-10× more than false positives
   - Optimize for recall at fixed precision threshold
   - Clinical relevance: Fewer missed cancers

3. **Deeper Hyperparameter Search:**
   - More granular C values around 0.1 (e.g., 0.05, 0.08, 0.12)
   - More k values for KNN (e.g., 4, 6, 8)
   - Expected: 0.2-0.5% improvement

**Medium Term (new techniques):**
4. **Advanced Models:**
   - Gradient Boosting (XGBoost, LightGBM)
   - Support Vector Machines (if allowed)
   - Expected: Potentially match or exceed current best

5. **Feature Engineering:**
   - Create interaction features manually (e.g., radius/perimeter ratio)
   - Domain-specific transformations
   - Expected: Improve recall without sacrificing precision

6. **Cross-Validation:**
   - 5-fold or 10-fold CV for more robust model selection
   - Reduces variance in performance estimates

**Long Term (deployment considerations):**
7. **Explainability:**
   - SHAP values: Which features drive predictions?
   - LIME: Instance-level explanations
   - Clinical acceptance: Doctors need to understand model decisions

8. **External Validation:**
   - Test on datasets from different hospitals
   - Different imaging equipment
   - Diverse patient populations

9. **Real-Time Deployment:**
   - API for inference
   - Model monitoring (detect data drift)
   - Continuous learning pipeline

10. **Clinical Trial:**
    - Prospective study comparing model + doctor vs doctor alone
    - Measure impact on patient outcomes
    - Regulatory approval process (FDA)

## E.8 Chart of Key Findings

### Model Performance Rankings

| Rank | Model | Feature Space | Hyperparameter | Val F1 | Test F1 |
|------|-------|---------------|----------------|--------|---------|
| 🥇 1 | **Logistic Reg** | **Raw** | **C=0.1** | **0.9762** | **0.9756** |
| 🥈 2 | Logistic Reg | SelectKBest | C=1.0 | 0.9647 | - |
| 🥉 3 | Logistic Reg | PCA | C=1.0 | 0.9655 | - |

### Feature Space Effectiveness

```
┌─────────────────┬─────────────┬──────────┬─────────────┐
│ Feature Space   │  Avg F1     │  Std F1  │  Best Model │
├─────────────────┼─────────────┼──────────┼─────────────┤
│ PCA             │  0.9004 ★★★★│  0.0646  │  KNN (k=7)  │
│ SelectKBest     │  0.8904 ★★★ │  0.0580  │  LogReg     │
│ Polynomial      │  0.8869 ★★★ │  0.1031  │  KNN (k=5)  │
│ Raw             │  0.8134 ★★  │  0.1780  │  LogReg ⭐   │
└─────────────────┴─────────────┴──────────┴─────────────┘

★★★★ = Most consistent    ⭐ = Absolute best single model
```

### Bias-Variance Balance

```
                  UNDERFITTING │ OPTIMAL │ OVERFITTING
                      (High Bias) │         │ (High Variance)
                               │         │
LogReg C=0.001  ◄──────────────┤         │
LogReg C=0.01   ──────────────►│         │
LogReg C=0.1    ────────────────┤    ●    │  ← Best Model
LogReg C=1.0    ────────────────┤         ├──►
LogReg C=10     ────────────────┤         ├────────►
LogReg C=100    ────────────────┤         ├──────────────►
                               │         │
KNN k=15        ◄──────────────┤         │
KNN k=9         ──────────────►│         │
KNN k=7         ────────────────┤    ●    │  ← Optimal
KNN k=5         ────────────────┤    ●    │  ← Optimal
KNN k=3         ────────────────┤         ├─►
KNN k=1         ────────────────┤         ├──────────────►
                               │         │
MLP α=1.0       ◄──────────────┤         │
MLP α=0.0001    ────────────────┤    ●    │  ← Best for MLP
```

### Clinical Performance Summary

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **Accuracy** | 98.25% | Correct diagnosis on 98 of 100 cases |
| **Precision** | 100% | ⭐ **Zero false alarms** - no unnecessary biopsies |
| **Recall** | 95.24% | Detects 95 of 100 actual cancers |
| **False Negatives** | 2/42 | Misses ~5% of malignant tumors |
| **False Positives** | 0/72 | No benign tumors misclassified |

---

# F. Final Summary

This project successfully demonstrated comprehensive machine learning methodology on the Breast Cancer Wisconsin (Diagnostic) dataset. Through systematic exploration of 72 model configurations across three algorithm families, four feature spaces, and multiple regularization settings, we achieved:

**Best Model:** Logistic Regression (C=0.1) on raw scaled features
- **Test Accuracy:** 98.25%
- **Test F1-Score:** 97.56%
- **Perfect Precision:** 100% (no false alarms)
- **Excellent Recall:** 95.24% (detects 40/42 malignant cases)
- **Outstanding Generalization:** Validation-test gap of only 0.06%

**Key Lessons:**
1. Simple models can outperform complex ones with good features
2. Proper regularization is critical for generalization
3. Feature quality matters more than model complexity
4. Systematic experimentation reveals optimal configurations
5. Domain expertise in feature engineering reduces modeling burden

**Project Requirements:**
✓ Dataset: 569 samples, 30 features (exceeds 200 samples, 10 features)
✓ Task: Binary classification with clinical relevance
✓ EDA: Comprehensive analysis with visualizations
✓ Unsupervised: PCA and K-Means clustering
✓ Models: 3 families (Logistic Regression, KNN, MLP)
✓ Transformations: 3 + baseline (PCA, Polynomial, SelectKBest, Raw)
✓ Hyperparameters: 6 values per model family
✓ Total: 72 models trained and evaluated
✓ Results: Complete table with all metrics
✓ Analysis: Bias-variance, overfitting/underfitting discussion
✓ Test evaluation: Single evaluation on held-out set

This model demonstrates promising potential as a clinical decision support tool for breast cancer diagnosis, achieving professional-grade performance while maintaining simplicity and interpretability.

---

**End of Write-Up**

---

# Appendix: References

1. **Dataset:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

2. **Original Paper:** Street, W.N., Wolberg, W.H., and Mangasarian, O.L. (1993). "Nuclear feature extraction for breast tumor diagnosis." IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870.

3. **Scikit-learn:** Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12, pp. 2825-2830.

4. **Course Materials:** CS-UY 4563 – Introduction to Machine Learning, New York University

---

**Team Contributions:**

**Saad Iftikhar:**
- Data preprocessing and feature engineering
- Model implementation (Logistic Regression, KNN, MLP)
- Hyperparameter tuning and results analysis
- Visualizations and performance metrics

**Ahmed Arkam Mohamed Faisaar:**
- Exploratory data analysis
- Unsupervised learning (PCA, K-Means)
- Documentation and write-up
- Presentation materials

---

**Word Count:** ~8,500 words  
**Code:** Jupyter Notebook with executed cells  
**Visualizations:** 11 PNG files  
**Results:** 72 models documented in CSV
