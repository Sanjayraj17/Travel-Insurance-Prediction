Below is a **README.md** you can directly use for this project. It explains the **methodology, dataset, preprocessing, EDA insights, and model selection** clearly and in an exam/project-submission friendly manner.

---

# Travel Insurance Prediction using Random Forest

## 📌 Project Overview

This project focuses on predicting whether a customer will purchase **Travel Insurance** using demographic, financial, and behavioral attributes. A **Random Forest Classifier** is used as the primary machine learning model, with performance further improved using **GridSearchCV** for hyperparameter tuning.

---

## 📊 Dataset Description

The dataset used is **TravelInsurancePrediction.csv**, which contains customer-related information relevant to travel insurance purchase decisions.

### 🔢 Features in the Dataset

| Feature Name        | Description                                       |
| ------------------- | ------------------------------------------------- |
| Age                 | Age of the customer                               |
| Employment Type     | Government sector or Private/Self-employed        |
| AnnualIncome        | Yearly income of the customer                     |
| FamilyMembers       | Number of family members                          |
| ChronicDiseases     | Whether the customer has chronic diseases (0/1)   |
| FrequentFlyer       | Whether the customer frequently travels (Yes/No)  |
| EverTravelledAbroad | Whether the customer has traveled abroad (Yes/No) |
| TravelInsurance     | Target variable (0 = No, 1 = Yes)                 |

### ❌ Dropped Columns

* **Unnamed: 0** – Index column with no predictive value
* **GraduateOrNot** – Removed to reduce redundancy and noise

---

## 🧹 Data Preprocessing Steps

1. **Missing Value Handling**

   * Checked using `df.isna().sum()`
   * No missing values were found, so no imputation was required.

2. **Feature Removal**

   * Irrelevant columns (`Unnamed: 0`, `GraduateOrNot`) were dropped.

3. **Categorical Encoding**

   * Binary categorical variables were encoded using mapping:

     ```python
     Yes → 1, No → 0
     ```
   * Employment Type:

     * Government Sector → 1
     * Private/Self Employed → 0

4. **Feature & Target Separation**

   * Independent variables stored in **X**
   * Target variable (`TravelInsurance`) stored in **y**

5. **Train-Test Split**

   * 80% Training Data
   * 20% Testing Data
   * `random_state = 42` for reproducibility

---

## 🔍 Exploratory Data Analysis (EDA)

Several visualizations were used to understand patterns and relationships:

### 📈 Correlation Heatmap

* Identified relationships between numerical variables.
* Helped detect multicollinearity.
* Annual income and travel-related features showed notable influence.

### 👥 Age Distribution

* Most customers fall within a specific working-age group.
* Useful for understanding insurance demand demographics.

### 👨‍👩‍👧 Family Members Distribution

* Smaller family sizes are more common.
* Helps identify family-based insurance trends.

### 💰 Annual Income Distribution

* Right-skewed distribution.
* Mean income marked to identify income concentration.
* No extreme outliers detected that required removal.

### 🩺 Chronic Diseases

* Majority of customers do not have chronic diseases.
* Customers with chronic diseases showed different insurance behavior.

### ✈️ Frequent Flyer Status

* Frequent flyers showed higher likelihood of purchasing travel insurance.

### 🧾 Travel Insurance Distribution

* Class distribution visualized using a pie chart.
* Slight class imbalance, but manageable without resampling.

---

## 🚫 Outlier Handling

* Income distribution was visually inspected using histograms.
* No extreme outliers observed.
* Therefore, no outlier removal or transformation was applied.

---

## 🤖 Model Methodology

### 🔹 Classification Model Used

**Random Forest Classifier**

### 🔹 Reason for Selecting Random Forest

* Handles both numerical and categorical features effectively.
* Robust against overfitting due to ensemble learning.
* Captures non-linear relationships.
* Works well without extensive feature scaling.
* Provides high accuracy for structured/tabular data.

---

## 📈 Model Training & Evaluation

### 🔸 Initial Model

* Random Forest trained with default parameters.
* Accuracy evaluated using test data.

### 🔸 Hyperparameter Tuning

**GridSearchCV** was applied to improve performance by tuning:

* Number of trees (`n_estimators`)
* Tree depth (`max_depth`)
* Splitting criteria (`gini`, `entropy`)
* Minimum samples for split and leaf
* Feature selection size

### 🔸 Evaluation Metrics

* Accuracy Score
* Precision
* Recall
* F1-score
* Classification Report

---

## 🏆 Results

* GridSearchCV improved the model performance compared to the base Random Forest.
* The final tuned model achieved better generalization and classification accuracy.
* Precision and recall values indicate reliable prediction of insurance purchase behavior.

---

## 🛠 Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

## ✅ Conclusion

This project successfully demonstrates how **Random Forest Classification**, combined with effective preprocessing and EDA, can be used to predict customer travel insurance purchase behavior. Hyperparameter tuning further enhances model accuracy and robustness.

---



Just tell me 👍
