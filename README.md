# SMARTCOURSEFINDER

## Overview

Each year, students applying for tertiary education through KUCCPS (Kenya Universities and Colleges Central Placement Service) face challenges in identifying suitable course clusters. These challenges often stem from the **dynamic nature of cluster point requirements**, **variation in student performance**, and **frequent changes in course offerings** across institutions.

Students frequently lack the tools or insights to make **data-driven decisions** about which courses they are likely to qualify for, leading to **suboptimal course applications** or even outright rejections.

---

# CRISP-DM Methodology

## 1. Business Understanding

### 🔹 Problem Statement
Students applying through KUCCPS face difficulty in selecting appropriate courses due to fluctuating cluster point cutoffs and institutional changes. Lack of insight leads to missed opportunities and frequent application rejections.

### 🔹 Project Objectives
- Build a **predictive model** to estimate probable course cluster cutoffs.
- Recommend **suitable programs** based on a student's subject performance.
- Provide a **data-driven recommendation system** to guide course selection.

---

## 2. Data Understanding

### 🔹 Dataset Overview
- **File**: `programmes.csv`
- **Entries**: 1275
- **Columns**: 22
- **Years covered**: 2015 to 2021
- **Fields**: Degree name, subject combinations, cutoff scores, institution, requirements

### 🔹 Sample Preview
```python
df = pd.read_csv("programmes.csv")
df.head() 
```
## 3. Data Preparation

### Format Correction
Initial inspection of the DataFrame's structure and data types.

```
df.info()
```
### Handling Missing Values
Imputation of missing values for the year columns (2015-2021) using forward and backward filling.

```
nan_years = [str(y) for y in range(2015, 2022)]
df[nan_years] = df[nan_years].ffill(axis=1)
df[nan_years] = df[nan_years].bfill(axis=1)
```
### Feature Engineering
Subjects were grouped into 5 categories (Core, Sciences, Humanities, Technical, Languages). A unique "Cluster Group" was assigned to each course based on its subject combination fingerprint.

```
# Create a unique key from the sorted subject combinations
df['subject_key'] = df[['Cluster Subject 1','Cluster Subject 2','Cluster Subject 3','Cluster Subject 4']].apply(
    lambda row: tuple(sorted(row.astype(str).str.strip())), axis=1
)

# Map each unique combination to a cluster group ID
subject_to_cluster = {combo: idx+1 for idx, combo in enumerate(df['subject_key'].unique())}
df['Cluster Group'] = df['subject_key'].map(subject_to_cluster)
```
## 4. Modeling
### Libraries Used
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
```
### Approach
Model: RandomForestRegressor

Input Features: Cluster group, subject combinations, institution

Target Output: Predicted cluster points for upcoming years

Goal: To predict future cutoff points based on historical data.

## 5. Evaluation
### Metrics Used
1. R² Score

2. Mean Squared Error (MSE)

Evaluation results will be added once the model is finalized and tested on validation data.

## 6. Deployment
Build a simple web interface where students can input their KCSE subject grades.

Use the trained model to recommend qualifying programs based on their predicted cluster scores.

Display historical trends, institution data, and cutoffs interactively.