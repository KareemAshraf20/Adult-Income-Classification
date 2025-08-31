# Adult Income Classification Project

## 📋 Project Overview
This project focuses on predicting whether an individual's income exceeds $50K/year based on census data using various machine learning classification algorithms. The dataset contains demographic and employment-related features that are used to build predictive models.

## 🎯 Objective
To develop and compare multiple machine learning models that can accurately classify individuals into income brackets (<=50K or >50K) based on their demographic and employment characteristics.

## 📊 Dataset
The dataset used is the **Adult Census Income dataset** (`adult.csv`), containing 32,561 instances with 15 features including:
- Demographic information (age, gender, race)
- Education level
- Occupation type
- Marital status
- Relationship status
- Work class
- Native country
- Financial information (capital gain/loss)
- Hours worked per week

## 🛠️ Technologies Used
- Python 3
- Pandas & NumPy (Data manipulation)
- Scikit-learn (Machine learning models)
- Matplotlib & Seaborn (Data visualization)

## 📝 Code Implementation

### 1. Importing Libraries
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import seaborn as sns
```

### 2. Data Loading and Exploration
```python
# Load the dataset with proper column names
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 
             'marital-status', 'occupation', 'relationship', 'race', 'gender', 
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
df = pd.read_csv('/content/adult.csv', sep=',', names=col_names, na_values="?", header=None)
```

### 3. Data Preprocessing
```python
# Remove irrelevant column
df.drop(columns=['fnlwgt'], inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)
```

### 4. Data Splitting
```python
# Separate features and target variable
x = df.drop(columns=['income_ >50K'])
y = df['income_ >50K']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 5. Model Training and Evaluation

#### K-Nearest Neighbors (KNN)
```python
kn = KNeighborsClassifier(n_neighbors=13)
kn.fit(x_train, y_train)
y_pred_kn = kn.predict(x_test)
print("The accuracy score is: ", accuracy_score(y_test, y_pred_kn)*100)
```

#### Logistic Regression
```python
Lo = LogisticRegression(max_iter=900)
Lo.fit(x_train, y_train)
y_pred_lo = Lo.predict(x_test)
accuracy_score(y_test, y_pred_lo)
```

#### Support Vector Classifier (SVC)
```python
svc = SVC(C=5)
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
print("The accuracy score is: ", accuracy_score(y_test, y_pred_svc)*100)
```

#### Naive Bayes
```python
model = GaussianNB()
model.fit(x, y)
print("Accuracy = ", accuracy_score(y, model.predict(x))*100)
```

## 📈 Results
The models achieved the following performance:
- **K-Nearest Neighbors**: 85.34% accuracy
- **Logistic Regression**: 85.20% accuracy  
- **Support Vector Classifier**: 81.07% accuracy
- **Naive Bayes**: 82.72% accuracy

## 📋 Key Findings
- KNN and Logistic Regression performed the best on this dataset
- The dataset required significant preprocessing due to categorical variables and missing values
- Feature engineering through one-hot encoding was crucial for model performance
- Class imbalance was present in the target variable (income brackets)
