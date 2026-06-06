# 🌾 Crop Yield Prediction and Resource Optimization Using Machine Learning and Linear Programming

## 📌 Project Overview

Agriculture is one of the most important sectors where crop productivity depends on several environmental, soil, and resource-related factors such as rainfall, temperature, humidity, soil pH, fertilizer usage, and water availability.

Traditional farming decisions are mainly based on experience, which may not always provide accurate yield estimation or efficient resource utilization.

This project develops a **data-driven smart agriculture system** by integrating:

- **Machine Learning (ML)** for crop yield prediction
- **Operational Research (OR)** using Linear Programming for resource optimization

The Machine Learning model predicts crop yield using historical agricultural data, while Linear Programming recommends optimal fertilizer and water usage to maximize profit and reduce resource wastage.

---

# 🎯 Project Objectives

## 1. Crop Yield Prediction

The main objective is to build a Machine Learning model that predicts crop yield based on agricultural and environmental conditions.

Example:

```
Input Features
      |
      ↓
Machine Learning Model
      |
      ↓
Predicted Crop Yield
```

The model learns relationships between factors like rainfall, temperature, soil condition, and fertilizer usage to estimate production.

---

## 2. Agricultural Factor Analysis

The project analyzes how different features influence crop yield.

Important factors:

- Rainfall
- Temperature
- Humidity
- Soil pH
- Fertilizer nutrients

Techniques used:

- Exploratory Data Analysis (EDA)
- Data Visualization
- Correlation Analysis

---

## 3. Handling Multicollinearity

Multicollinearity occurs when independent variables are highly correlated with each other.

Example:

```
Nitrogen (N)
Phosphorus (P)
Potassium (K)
```

Highly related variables can affect model performance.

To solve this problem:

- Correlation analysis was performed
- Variance Inflation Factor (VIF) was used
- Unnecessary features were removed

---

## 4. Resource Optimization

Prediction alone does not help farmers make complete decisions.

After predicting yield, Linear Programming answers:

"How much fertilizer and water should be used for maximum profit?"

Optimization provides:

- Best fertilizer quantity
- Best water allocation
- Maximum achievable profit

---

# 📊 Dataset Description

The dataset contains approximately **50,000 agricultural records** across different regions.

Each record contains crop, environmental, and resource information.

---

## Dataset Features

| Feature | Description |
|---|---|
| Year | Crop cultivation year |
| State Name | State where crop is grown |
| District Name | Cultivation district |
| Crop | Type of crop |
| Area | Farming area in hectares |
| Temperature | Average temperature |
| Humidity | Moisture percentage |
| Rainfall | Rain received in mm |
| Soil pH | Soil acidity/alkalinity |
| Wind Speed | Wind measurement |
| Solar Radiation | Sunlight energy |
| Nitrogen (N) | Nitrogen fertilizer requirement |
| Phosphorus (P) | Phosphorus requirement |
| Potassium (K) | Potassium requirement |

---

## Target Variable

The value predicted by the model:

```
Yield_kg_per_ha
```

Meaning:

Crop production obtained per hectare.

---

# 🛠️ Technologies Used

- Python
- Machine Learning
- Statistics
- Data Science
- Operational Research
- Linear Programming

---

# 📚 Python Libraries

## Pandas

Used for data handling and manipulation.

Example:

```python
df.head()

df.info()
```

---

## NumPy

Used for numerical calculations and array operations.

---

## Matplotlib & Seaborn

Used for data visualization.

Created:

- Histogram
- Boxplot
- Heatmap
- Scatter plot

---

## Scikit-Learn

Used for Machine Learning implementation:

- Data splitting
- Model training
- Prediction
- Evaluation

---

## SciPy

Used for Linear Programming optimization.

Example:

```python
linprog()
```

---

# 🔄 Project Workflow

# 1. Data Understanding

Initial dataset analysis was performed.

Checked:

- Number of records
- Column information
- Data types
- Statistical summary


```python
df.info()

df.describe()
```

---

# 2. Data Cleaning

Data preprocessing steps:

### Missing Values

Checked missing data:

```python
df.isnull().sum()
```

Missing values can reduce model performance.

---

### Duplicate Records

Checked duplicate rows:

```python
df.duplicated().sum()
```

Duplicate data can create biased models.

---

### Column Formatting

Cleaned column names:

```python
df.columns = df.columns.str.strip()
```

This improves code readability.

---

# 📈 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and relationships inside the dataset.

---

## Univariate Analysis

Analysis of a single variable.

Example:

Understanding rainfall distribution.

Visualization:

- Histogram
- Count plot

---

## Bivariate Analysis

Study relationship between two variables.

Example:

```
Rainfall vs Crop Yield
```

It helps understand how one factor affects another.

---

## Correlation Analysis

Correlation measures relationships between numerical variables.

Range:

```
-1 to +1
```

Meaning:

- +1 → Strong positive relationship
- 0 → No relationship
- -1 → Strong negative relationship


Heatmap was used for visualization.

---

# ⚙️ Feature Engineering

Feature engineering improves model performance by creating better input features.

---

## Total Fertilizer Feature

Combined fertilizer nutrients:

```python
Total_Fertilizer = N + P + K
```

This represents total fertilizer requirement.

---

## Outlier Handling

Outliers are extreme values that affect model accuracy.

Example:

Normal:

```
3000 kg
4000 kg
5000 kg
```

Outlier:

```
50000 kg
```

Handled using:

```
Interquartile Range (IQR)
```

---

## Encoding Categorical Data

Machine Learning models cannot directly understand text.

Converted categorical values into numerical format.

Example:

Before:

```
Rice
Cotton
```

After encoding:

```
Rice = 1
Cotton = 0
```

Technique:

```
One-Hot Encoding
```

---

## Feature Scaling

Scaling converts variables into similar ranges.

Example:

Before:

```
Rainfall = 1000
pH = 6
```

After:

```
Rainfall = 0.8
pH = 0.5
```

---

# 🤖 Machine Learning Models

Different regression models were developed and compared.

---

## Linear Regression

A baseline model that identifies linear relationships between input variables and crop yield.

---

## Ridge Regression

Improved regression model using L2 regularization.

Purpose:

- Reduce overfitting
- Improve generalization

---

## Lasso Regression

Uses L1 regularization.

Advantages:

- Feature selection
- Removes less important variables
- Controls overfitting

**Lasso Regression was selected as the best-performing model.**

---

## Random Forest Regressor

An ensemble model using multiple decision trees.

Advantages:

- Handles complex patterns
- Captures non-linear relationships

---

# 📊 Model Evaluation

Models were evaluated using:

## MAE

Mean Absolute Error

Measures average prediction error.

Lower value indicates better performance.

---

## MSE

Mean Squared Error

Measures squared difference between actual and predicted values.

---

## RMSE

Root Mean Squared Error

Shows prediction error in original units.

---

## R² Score

Measures model accuracy.

```
Closer to 1 = Better Model
```

---

# 📐 Linear Programming Optimization

After prediction, the ML output is connected with an optimization model.

The goal is:

```
Maximize Profit = Revenue - Resource Cost
```

---

## Decision Variables

Variables controlled by the optimization model:

- Fertilizer quantity
- Water usage

---

## Constraints

Real-world limitations:

Examples:

Budget:

```
Cost ≤ Available Budget
```

Water:

```
Water Usage ≤ Available Water
```

---

# 🌱 Final Output

The final system provides:

✔ Predicted crop yield

✔ Optimal fertilizer recommendation

✔ Optimal water allocation

✔ Maximum profit estimation

---

# 📌 Conclusion

This project combines Machine Learning and Linear Programming to create a smart agriculture decision-support system.

The Machine Learning model predicts crop productivity, while optimization techniques recommend efficient resource allocation.

This integrated ML + OR approach helps:

- Increase agricultural productivity
- Reduce farming cost
- Improve resource utilization
- Support sustainable farming practices

---

# 👨‍💻 Author

**Mali Satish**  
