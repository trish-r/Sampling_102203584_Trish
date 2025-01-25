# Predictive Sampling Analysis

## Overview
This project evaluates the performance of different machine learning models using various sampling techniques on a dataset. The accuracy of each model is compared across five distinct samples.

## Author Information
- **Name**: Trish Rustagi
- **Roll Number**: 102203584  

## Machine Learning Models
The following models were used in this project:
1. **Logistic Regression**
2. **Decision Tree**
3. **Ridge Regression**
4. **Linear Regression**
5. **k-Nearest Neighbors (k-NN)**

## Sampling Methods
Five different sampling techniques were applied to the dataset:
1. **Random Sampling**
2. **Stratified Sampling**
3. **Systematic Sampling**
4. **Cluster Sampling**
5. **Bootstrap Sampling**

## Results
Below is the model performance (accuracy) for each sample:

| Model                 | Sample1 | Sample2 | Sample3 | Sample4 | Sample5 |
|-----------------------|---------|---------|---------|---------|---------|
| **Logistic Regression** | 0.885246 | 0.951613 | 0.901639 | 0.758065 | 0.950820 |
| **Decision Tree**      | 0.918033 | 0.967742 | 0.950820 | 0.790323 | 0.950820 |
| **Ridge Regression**   | 0.836066 | 0.903226 | 0.885246 | 0.774194 | 0.918033 |
| **Linear Regression**  | 0.819672 | 0.903226 | 0.885246 | 0.790323 | 0.918033 |
| **k-NN**               | 0.655738 | 0.741935 | 0.803279 | 0.725806 | 0.754098 |

## Steps to Reproduce

### Step 1: Import Required Libraries
```
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
```

### Step 2: Prepare Sampling
# Code for generating random, stratified, systematic, cluster, and bootstrap samples
# Refer to the project code for detailed implementation.

### Step 3: Initialize Models
```
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Linear Regression": LinearRegression(),
    "k-NN": KNeighborsClassifier(n_neighbors=5)
}
```

### Step 4: Evaluate Model Performance
```
# Iterate through models and samples to compute accuracy scores
performance_metrics = {}
for model_name, model in models.items():
    performance_metrics[model_name] = []
    for sample in samples:  # samples = [sample1, sample2, ...]
        X = sample.drop("Class", axis=1)
        y = sample["Class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        performance_metrics[model_name].append(accuracy)
```
### Save Results
```
results_table = pd.DataFrame(performance_metrics, index=["Sample1", "Sample2", "Sample3", "Sample4", "Sample5"])
results_table.to_csv("model_accuracy.csv")
```
