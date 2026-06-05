# 🌾 Crop Yield Prediction and Resource Optimization  
## Using Machine Learning and Linear Programming

---

## 📌 Complete Project Workflow

This project follows a complete **Data Science + Optimization pipeline**.  
The workflow starts from agricultural data understanding and ends with an intelligent decision-support system that predicts crop yield and optimizes farming resources.

---

## 📂 Project Structure

```text
🌾 Crop Yield Prediction and Resource Optimization

│
├── 📌 1. Project Introduction
│       │
│       └── 🎯 What problem we solved
│              ├── Crop yield prediction challenge
│              ├── Efficient fertilizer utilization
│              ├── Water resource management
│              └── Profit maximization for agriculture
│

├── 📊 2. Dataset Understanding
│       │
│       ├── 📁 What data we used
│       │      ├── Historical crop production data
│       │      ├── Climate information
│       │      ├── Soil properties
│       │      └── Fertilizer requirements
│       │
│       ├── 🌱 Why each feature matters
│       │      ├── Temperature effect on crop growth
│       │      ├── Rainfall contribution
│       │      ├── Soil pH importance
│       │      ├── N-P-K fertilizer impact
│       │      └── Environmental influence
│       │
│       └── 🐍 Python Implementation
│              ├── Dataset loading
│              ├── Data inspection
│              └── Statistical analysis
│


├── 🧹 3. Data Cleaning
│       │
│       ├── 🔍 Missing Values
│       │      ├── Check null values
│       │      └── Handle incomplete records
│       │
│       ├── 📌 Duplicate Records
│       │      ├── Identify duplicates
│       │      └── Remove repeated information
│       │
│       └── ⚙ Why preprocessing required
│              ├── Improve data quality
│              ├── Reduce errors
│              └── Increase model accuracy
│


├── 📈 4. Exploratory Data Analysis (EDA)
│       │
│       ├── 📊 What patterns we analyzed
│       │      ├── Feature distributions
│       │      ├── Crop production trends
│       │      ├── Climate relationships
│       │      └── Yield behavior
│       │
│       ├── 📉 Visualization Code
│       │      ├── Histogram
│       │      ├── Boxplot
│       │      ├── Heatmap
│       │      └── Scatter plots
│       │
│       └── 💡 Insights Obtained
│              ├── Important variables
│              ├── Feature relationships
│              └── Data patterns
│


├── 📦 5. Outlier Handling
│       │
│       ├── 📐 IQR Theory
│       │      ├── Q1 and Q3 calculation
│       │      ├── IQR = Q3 - Q1
│       │      └── Outlier boundary detection
│       │
│       ├── 🐍 Implementation
│       │      ├── Detect extreme values
│       │      └── Apply capping method
│       │
│       └── 🎯 Why Capping Used
│              ├── Preserve information
│              ├── Reduce noise
│              └── Improve model stability
│


├── 🛠 6. Feature Engineering
│       │
│       ├── 🌱 Total Fertilizer Creation
│       │      ├── Nitrogen (N)
│       │      ├── Phosphorus (P)
│       │      ├── Potassium (K)
│       │      └── Total Fertilizer = N + P + K
│       │
│       ├── ❓ Why Created
│       │      ├── Reduce redundancy
│       │      ├── Control multicollinearity
│       │      └── Improve feature quality
│       │
│       └── 🐍 Code Explanation
│              ├── Feature creation
│              └── Data transformation
│


├── 🔎 7. Feature Selection
│       │
│       ├── 🔥 Correlation Analysis
│       │      ├── Relationship checking
│       │      └── Heatmap visualization
│       │
│       ├── 📊 Variance Inflation Factor (VIF)
│       │      ├── Detect dependency
│       │      └── Identify redundant variables
│       │
│       └── ⚙ Multicollinearity Removal
│              ├── Remove highly related features
│              └── Improve model interpretation
│


├── 🤖 8. Model Development
│       │
│       ├── 📈 Linear Regression
│       │      └── Baseline prediction model
│       │
│       ├── 📉 Ridge Regression
│       │      ├── L2 Regularization
│       │      └── Reduce overfitting
│       │
│       ├── ⭐ Lasso Regression
│       │      ├── L1 Regularization
│       │      ├── Feature selection
│       │      └── Best performing model
│       │
│       └── 🌲 Random Forest
│              ├── Ensemble learning
│              └── Non-linear prediction
│


├── 📊 9. Model Evaluation
│       │
│       ├── 📌 R² Score
│       │      └── Prediction accuracy
│       │
│       ├── 📌 MAE
│       │      └── Average prediction error
│       │
│       ├── 📌 RMSE
│       │      └── Error magnitude checking
│       │
│       └── 📌 Cross Validation
│              └── Model generalization testing
│


└── 🚀 10. Resource Optimization
        │
        ├── 🔗 ML + OR Connection
        │      ├── ML predicts crop yield
        │      └── OR optimizes resources
        │
        ├── 📈 Linear Programming
        │      └── Mathematical optimization model
        │
        ├── 🎯 Decision Variables
        │      ├── Fertilizer quantity
        │      └── Water allocation
        │
        ├── 📌 Constraints
        │      ├── Budget limitation
        │      ├── Resource availability
        │      └── Environmental restrictions
        │
        └── 💰 Profit Maximization
               ├── Maximum yield
               ├── Minimum resource wastage
               └── Better agricultural planning

```

---





## 📌 Project Overview

### What we did?

In this project, we developed a smart agricultural decision-support system by combining Machine Learning and Operational Research.

The system performs two major tasks:

1. Predicts crop yield using historical agricultural data

2. Optimizes fertilizer and water resources to maximize farmer profit


Instead of only predicting crop production, this project converts prediction into a decision-making solution.


---

# 📂 Step 1: Dataset Loading and Understanding


## What we did?

The first step was loading agricultural historical data and understanding the structure of the dataset.

The dataset contains information about:

- Crop details
- Location
- Climate conditions
- Soil properties
- Fertilizer requirements


Target Variable:

```
Yield_kg_per_ha
```


This represents the amount of crop produced per hectare.


---

## Implementation


```python
import pandas as pd

df = pd.read_csv(
"Custom_Crops_yield_Historical_Dataset.csv"
)

df.head()
```


---

## Code Explanation


### pandas

Used for reading and handling tabular data.


### read_csv()

Loads CSV dataset into DataFrame format.


### df.head()

Displays first five rows to understand:

- Column names
- Data format
- Feature values


---

# Dataset Information Checking


## What we did?


Before model building, we analyzed:

- Number of rows
- Number of columns
- Datatypes
- Statistical information


## Implementation


```python
df.info()

df.describe()

df.shape
```


## Explanation


### df.info()

Shows:

- Column datatype
- Missing values
- Memory usage


### df.describe()

Provides:

- Mean
- Standard deviation
- Minimum value
- Maximum value


This helped us understand feature distribution.


---

# 🧹 Step 2: Data Cleaning


## What we did?


Machine learning requires clean data.

Therefore we checked:

- Missing values
- Duplicate records


---

# Missing Value Detection


## Implementation


```python
df.isnull().sum()
```


## Explanation


Missing values can affect model learning.

If missing values exist:

- Replace with mean/median

or

- Remove records


In our dataset:

```
No missing values found
```


---

# Duplicate Checking


## Implementation


```python
df.duplicated().sum()
```


## Explanation


Duplicate records create biased learning because the model sees repeated examples.

Result:

```
No duplicate records found
```


---

# 📊 Step 3: Exploratory Data Analysis


## What we did?


EDA was performed to understand hidden relationships inside agricultural data.


We studied:


### Crop behavior

### Weather effect

### Fertilizer impact

### Yield patterns


---

# Correlation Analysis


## Why?


Correlation tells how strongly features are related.


Example:


If Nitrogen and Phosphorus give almost same information, keeping both increases complexity.


---

## Implementation


```python
correlation = df.corr(
numeric_only=True
)

sns.heatmap(
correlation,
annot=True
)
```


---

## Explanation


Heatmap helped identify:

- Important features

- Highly correlated variables

- Redundant information


---

# ⚙️ Step 4: Feature Engineering


## What we did?


Created a new meaningful variable:


```
Total Fertilizer
```


using:


Nitrogen + Phosphorus + Potassium


---

## Why?


During correlation analysis:

N, P and K showed high relationship.


Keeping all three causes:


```
Multicollinearity
```


So we combined them.


---

## Implementation


```python
df["Total_Fertilizer"] = (

df["N_req_kg_per_ha"]

+

df["P_req_kg_per_ha"]

+

df["K_req_kg_per_ha"]

)
```


---

## Benefits


✔ Reduces model complexity

✔ Removes repeated information

✔ Improves prediction reliability


---

# 🔥 Step 5: Model Training


## What we did?


We trained multiple regression algorithms.


Because crop yield prediction is a:

```
Regression Problem
```


Output is continuous numeric value.


---

# Linear Regression


## Theory


Finds relationship:


Input Features → Crop Yield


Equation:


Y = β0 + β1X1 + β2X2 + error


---

## Implementation


```python
lr = LinearRegression()

lr.fit(
X_train,
y_train
)
```


## Explanation


fit()

means model learns patterns from training data.


---

# Lasso Regression


## What we did?


Applied L1 regularization model.


---

## Why?


Lasso:

- Reduces overfitting

- Selects important features

- Removes unnecessary variables


---

## Implementation


```python
lasso = Lasso()

lasso.fit(
X_train,
y_train
)
```


---

# 📈 Step 6: Model Evaluation


## What we did?


Checked model performance using:


- R² Score
- MAE
- RMSE


---

## Implementation


```python
r2_score(
y_test,
prediction
)

mean_absolute_error(
y_test,
prediction
)
```


---

## Explanation


### R² Score

Higher value means better prediction.


### MAE

Average difference between actual and predicted yield.


### RMSE

Shows prediction error magnitude.


---

# 🌱 Step 7: Resource Optimization


## What we did?


After prediction, we connected ML output with Linear Programming.


ML answers:


"What will be the yield?"


Optimization answers:


"What resources should be used to get maximum profit?"


---

# Linear Programming Model


## Objective


Maximize:


```
Profit = Revenue - Cost
```


---

# Decision Variables


Variables controlled by farmer:


```
Fertilizer Amount

Water Quantity
```


---

# Implementation


```python
problem = LpProblem(
"Crop Optimization",
LpMaximize
)


fertilizer = LpVariable(
"fertilizer",
lowBound=0
)


water = LpVariable(
"water",
lowBound=0
)
```


---

## Explanation


LpProblem creates optimization model.


LpVariable creates values that the algorithm will optimize.


---

## 🎯 Final Outcome

The developed system provides:

🌾 **Accurate Crop Yield Prediction**

🌱 **Optimal Fertilizer Recommendation**

💧 **Efficient Water Allocation**

💰 **Maximum Profit Estimation**

This project demonstrates how **Machine Learning + Operational Research** can be integrated to develop an intelligent smart agriculture decision-support system.

# Project Conclusion


This project successfully combines:

Machine Learning + Linear Programming


Machine Learning provides prediction capability.

Optimization provides decision-making capability.


Together they create a smart agriculture system for improving productivity and efficient resource management.

# 👨‍💻 Author

## Mali Satish


        
