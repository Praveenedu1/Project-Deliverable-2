# Student Performance Regression Analysis

## Project Overview
This project implements regression modeling techniques to predict student final grades (G3) using various student demographic, social, and academic features. The analysis compares the performance of Linear Regression, Ridge Regression, and Lasso Regression models to identify the best predictor of student academic outcomes.

## Dataset Summary
- **Dataset**: Student Performance Dataset
- **Size**: 395 students × 33 features
- **Target Variable**: G3 (final grade, ranging from 0-20)
- **Features**: Mix of categorical and numerical variables including:
  - **Demographic**: school, sex, age, address, family size
  - **Social**: parent education, family relationships, going out frequency
  - **Academic**: study time, past failures, absences, support systems
  - **Behavioral**: alcohol consumption, health status

### Key Dataset Characteristics:
- **No missing values** - Complete dataset
- **Target distribution**: Final grades (G3) show a roughly normal distribution with mean ≈ 10.4
- **High correlation** between G1, G2, and G3 (previous grades strongly predict final grades)
- **Balanced categorical features** across most demographic variables

## Modeling Process

### 1. Data Preprocessing
- **Feature Selection**: Excluded G1 and G2 (previous grades) to avoid data leakage
- **Feature Engineering**: 
  - Standardized numerical features using StandardScaler
  - One-hot encoded categorical variables with `drop='first'` to avoid multicollinearity
- **Train-Test Split**: 80-20 split with random_state=42 for reproducibility

### 2. Model Implementation
Three regression models were implemented using scikit-learn pipelines:

#### Linear Regression
- Standard ordinary least squares regression
- No regularization
- Baseline model for comparison

#### Ridge Regression (L2 Regularization)
- Alpha = 1.0
- Penalizes large coefficients to reduce overfitting
- Maintains all features but shrinks coefficients

#### Lasso Regression (L1 Regularization)
- Alpha = 0.1
- Performs feature selection by driving some coefficients to zero
- Creates sparse models

### 3. Model Evaluation
Models were evaluated using:
- **R-squared (R²)**: Proportion of variance explained
- **Mean Squared Error (MSE)**: Average squared prediction errors
- **Root Mean Squared Error (RMSE)**: Square root of MSE, interpretable in original units
- **5-fold Cross-Validation**: To assess generalization capability

## Results and Evaluation

### Test Set Performance
| Model | R² | MSE | RMSE |
|-------|----|----|------|
| **Linear Regression** | 0.1415 | 17.6037 | 4.1957 |
| **Ridge Regression** | 0.1437 | 17.5576 | 4.1902 |
| **Lasso Regression** | 0.1306 | 17.8274 | 4.2223 |

### Cross-Validation Results
| Model | CV R² (mean ± std) |
|-------|--------------------|
| Linear Regression | 0.0015 ± 0.0958 |
| Ridge Regression | 0.0142 ± 0.0934 |
| **Lasso Regression** | **0.0641 ± 0.0663** |

## Key Insights and Observations

### Model Performance Analysis
1. **Best Overall Model**: Ridge Regression performed best on the test set with highest R² (0.1437) and lowest RMSE (4.1902)

2. **Cross-Validation Winner**: Lasso Regression showed the most consistent performance across folds with highest CV R² (0.0641) and lowest standard deviation (0.0663)

3. **Limited Predictive Power**: All models achieved relatively low R² scores (≤0.14), indicating that the selected features explain only about 14% of the variance in final grades

4. **Feature Importance**: The exclusion of G1 and G2 significantly reduced model performance, suggesting that previous academic performance is the strongest predictor of final grades

### Data Insights
- **Grade Correlation**: Strong correlations between G1, G2, and G3 (>0.8) indicate consistent academic performance patterns
- **Study Habits**: Study time and past failures show moderate correlations with final grades
- **Social Factors**: Family relationships, going out frequency, and alcohol consumption have weaker but measurable impacts

## Challenges Encountered and Solutions

### 1. **Low Model Performance**
- **Challenge**: All models achieved low R² scores without G1 and G2
- **Solution**: This was intentional to avoid data leakage, representing a realistic scenario where we predict final grades without knowing intermediate grades

### 2. **OneHotEncoder Warnings**
- **Challenge**: "Unknown categories" warnings during cross-validation
- **Solution**: Used `handle_unknown='ignore'` parameter to handle unseen categories gracefully

### 3. **Feature Selection Trade-off**
- **Challenge**: Balancing predictive power with realistic prediction scenarios
- **Solution**: Excluded highly correlated grade features (G1, G2) to create a more practical model that could be used early in the academic term

### 4. **Model Overfitting Concerns**
- **Challenge**: Ensuring models generalize well to unseen data
- **Solution**: Implemented regularization techniques (Ridge, Lasso) and cross-validation to assess generalization capability

## Technical Implementation

### Dependencies
```python
pandas, numpy, matplotlib, seaborn, scikit-learn
```

### Key Code Features
- **Modular Pipeline Design**: Preprocessing and modeling combined in sklearn pipelines
- **Reproducible Results**: Fixed random seeds for consistent outputs
- **Comprehensive Evaluation**: Multiple metrics and cross-validation
- **Visualization**: Performance comparison charts and correlation analysis

## Conclusions 

1. **Ridge Regression** is recommended for this dataset due to its superior test performance and stability
2. **Feature Engineering Opportunities**: Consider creating interaction terms or polynomial features to improve model performance
3. **Additional Data**: Collecting more behavioral and academic engagement features could improve prediction accuracy
4. **Real-world Application**: The model provides modest but meaningful insights for early identification of at-risk students


---
