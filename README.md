#  AllLife Bank – Personal Loan Campaign Analysis

##  Project Overview

A machine learning classification project for **AllLife Bank** to predict whether liability customers will accept personal loan offers based on their demographic and financial attributes. Uses Decision Tree algorithms with pre-pruning and post-pruning techniques to optimize customer targeting and campaign effectiveness while handling class imbalance in the dataset.

### Business Objective
> Convert liability customers (depositors) into asset customers (borrowers), while retaining them as depositors.

This project helps the bank identify potential liability customers who are likely to accept personal loans, so they can be targeted more effectively in future campaigns, improving the previous campaign's **9% success rate**.

##  Key Insights

###  Top Predictive Features:
| Feature | Importance | Key Finding |
|---------|-----------|-------------|
| **Income** | 59.37% | Highest predictor - $92.5K threshold identified |
| **Education_2** | 13.68% | Graduate/Advanced degree holders more likely (13.0%/13.7% vs 4.4% undergrad) |
| **CCAvg** | 7.85% | Credit card usage $2.95K monthly spending as secondary filter |
| **Family** | High | Smaller families (≤2) show highest conversion rates |
| **ZIPCode** | Regional | ZIP codes 92, 94, 93 show higher acceptance rates |

### Key Business Insights Discovered:
- **Income Threshold**: $92.5K annual income as primary decision criterion
- **Credit Card Behavior**: $2.95K monthly CCAvg spending indicates financial activity
- **Family Size Impact**: Smaller families (≤2) show highest conversion rates
- **Education Correlation**: Graduate (13.0%) and Advanced degree holders (13.7%) prefer loans vs undergraduates (4.4%)
- **Geographic Patterns**: Regional targeting opportunities in specific ZIP codes
- **Digital Engagement**: Online banking users more likely to accept loans

## Complete Architecture

### ML Classification Pipeline
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Customer      │    │  Data            │    │   Decision Tree │
│   Data (5000)   │───►│  Preprocessing   │───►│   Models        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Feature        │             │
         │              │  Engineering    │             │
         │              └─────────────────┘             │
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Model          │             │
         │              │  Optimization   │             │
         │              └─────────────────┘             │
         │                                               │
         └──────────────────────┬──────────────────────┘
                                │
                    ┌──────────▼──────────┐
                    │  Pruning & Model    │
                    │  Selection          │
                    └─────────────────────┘
```

## Complete Tech Stack

### Machine Learning Framework (Verified Implementation)
- **Data Processing:** pandas 1.5.3, numpy 1.25.2 for data manipulation
- **Machine Learning:** scikit-learn 1.2.2 for Decision Tree algorithms
- **Visualization:** matplotlib 3.7.1, seaborn 0.13.1 for EDA and analysis
- **Development Environment:** Google Colab with specific version controls

### Classification Algorithms & Techniques
- **Primary Algorithm:** DecisionTreeClassifier with Gini impurity criterion
- **Optimization:** Pre-pruning (max_depth, max_leaf_nodes, min_samples_split)
- **Advanced Pruning:** Cost-complexity pruning (ccp_alpha optimization)
- **Class Imbalance:** class_weight='balanced' for handling 9.6% positive class (highly imbalanced)
- **Model Selection:** Recall-focused evaluation for business optimization

##  Dataset Overview

- **Total Rows**: 5,000 customers
- **Total Columns**: 14 features
- **Target Variable**: `Personal_Loan` (binary: 0 = No, 1 = Yes)
- **Class Imbalance**: Only ~9.6% of customers accepted the loan → highly imbalanced dataset
- **Data Source**: AllLife Bank customer database

###  Feature List:

| Type | Features | Description |
|------|----------|-------------|
| **Numerical** | Age, Income, CCAvg, Family, Mortgage, Experience | Continuous variables |
| **Categorical** | Education, ZIPCode, CD_Account, Securities_Account, Online, CreditCard | Discrete variables |
| **Target** | Personal_Loan | 0 = Rejected, 1 = Accepted |
| **Dropped** | ID | Unique identifier (not predictive) |
| **Dropped** | Experience | Perfect correlation with Age (multicollinearity)

## Skills Developed

### Advanced Machine Learning & Data Science (Verified Implementation)
- **Decision Tree Mastery:** Gini impurity criterion, tree visualization, feature importance analysis
- **Model Optimization:** Hyperparameter tuning with pre-pruning (max_depth=2, max_leaf_nodes=50, min_samples_split=10)
- **Cost-Complexity Pruning:** Advanced post-pruning with ccp_alpha optimization (best_alpha=0.000272)
- **Class Imbalance Handling:** Weighted classes for 9.6% positive class distribution

### Feature Engineering & Data Analysis
- **Data Preprocessing:** Negative value correction (-1, -2, -3 → 1, 2, 3), ZIPCode transformation (467 → 7 regions)
- **One-Hot Encoding:** Categorical variables (ZIPCode, Education) with drop_first=True
- **Outlier Analysis:** IQR method for Income (1.92%), CCAvg (6.48%), Mortgage (5.82%) outliers
- **Feature Importance:** Income (59.37%), Education_2 (13.68%), CCAvg (7.85%) as top predictors

### Banking & Financial Domain Expertise
- **Personal Loan Analytics:** Converting liability customers (depositors) to asset customers (borrowers)
- **Customer Profiling:** Income thresholds ($92.5K), family size patterns, education correlation
- **Business Rules Generation:** Decision tree interpretation for marketing teams
- **Campaign Optimization:** High recall (84.56%) to minimize missed opportunities

## Technical Achievements (Verified Implementation)

##  Modeling Approach

Used **Decision Tree Classifier** due to its:
- Interpretability for business stakeholders
- Ability to handle class imbalance via `class_weight='balanced'`
- Generation of actionable business rules for marketing teams
- Non-linear decision boundary capability

###  Baseline Model Performance (Training):
| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Recall | 100% |
| Precision | 100% |
| F1-score | 100% |

>  **Overfitting Warning**: Perfect scores indicate overfitting due to deep tree structure - pruning required

---

###  Pre-Pruned Decision Tree (Hyperparameter Tuned)

**Hyperparameters:**
```python
max_depth=2, max_leaf_nodes=50, min_samples_split=10, class_weight='balanced'
```

| Metric | Training | Test |
|--------|----------|------|
| Accuracy | 79.0% | 77.9% |
| Recall | 100% | 100% |
| Precision | 31.1% | 31.0% |
| F1-score | 47.4% | 47.4% |

 **Perfect recall achieved** – no missed opportunities (no false negatives)
 Lower precision suggests many false positives – acceptable if outreach cost is low

---

###  Post-Pruned Decision Tree (Cost-Complexity Pruning)

**Optimization Method:** Cost-complexity pruning with `ccp_alpha=0.000272`

**Final Model Performance:**
| Metric | Training | Test |
|--------|----------|------|
| **Accuracy** | 99.97% | **97.8%** |
| **Recall** | 100% | **84.56%** |
| **Precision** | 99.70% | **92.65%** |
| **F1-score** | 99.85% | **88.42%** |

 **Achieves excellent balance** between recall and precision
 **High recall (84.56%)** ensures minimal missed opportunities (false negatives)
 **High precision (92.65%)** reduces unnecessary marketing costs (false positives)
 **Simple decision rules** can be shared with marketing teams for CRM segmentation

---

### Dataset & Business Context
- **Data Scale:** 5,000 customers with 14 features from AllLife Bank
- **Target Variable:** Personal_Loan acceptance (9.6% positive class - highly imbalanced)
- **Business Problem:** Convert liability customers to personal loan customers while retaining deposits
- **Performance Metric:** 9% success rate from previous campaign - goal to improve targeting

##  Best Performing Model

**Post-pruned Decision Tree** was selected as the final model because:
-  Balances model complexity and generalization
-  Maintains high recall (84.56%) - minimizes missed opportunities
-  Has significantly better precision (92.65%) than pre-pruned models
-  Is interpretable by business stakeholders and marketing teams
-  Generates actionable decision rules for customer segmentation
-  Achieves **97.8% accuracy** on test data

### Model Performance Results
- **Final Model:** Post-pruned Decision Tree with ccp_alpha=0.000272
- **Test Performance:** Accuracy=97.8%, Recall=84.56%, Precision=92.65%, F1-score=88.42%
- **Business Impact:** High recall ensures minimal missed opportunities (false negatives)
- **Operational Efficiency:** High precision reduces unnecessary marketing costs (false positives)
- **Class Weights:** {0: 0.15, 1: 0.85} to handle 9.6% positive class imbalance

##  Final Decision Rules (Simplified)

From the pruned tree - these rules can guide manual targeting and CRM segmentation:

```
Business Rules Extracted:
|--- Income <= 92.5
|    |--- CCAvg <= 2.95 → Definitely no (reject)
|    |--- CCAvg > 2.95 → Maybe yes (consider)
|--- Income > 92.5
     |--- Family <= 2 → Definitely yes (high priority)
     |--- Family > 2 → Probably yes (medium priority)
```

### Actionable Marketing Segments:

| Segment | Likelihood to Accept Loan | Recommended Strategy |
|---------|---------------------------|---------------------|
| **Income > $92.5K + Family ≤ 2** | Very High | Top priority for outreach |
| **Income > $92.5K + Family > 2** | High | Include in campaigns |
| **Income < $92.5K + CCAvg > $2.95K** | Medium-High | Target selectively |
| **Income < $92.5K + CCAvg < $2.95K** | Low | Avoid unless cost-effective |
| **CD Account Holder** | Medium | Cross-sell via relationship managers |
| **Graduate/Advanced Degree** | Medium | Use targeted content (13.0%/13.7% acceptance) |
| **ZIPCode 92/94/93** | Medium-High | Geographic promotions |
| **Online Banking Users** | Medium-High | Digital campaign focus |

### Key Business Insights Discovered
- **Income Threshold:** $92.5K annual income as primary decision criterion
- **Credit Card Behavior:** $2.95K monthly CCAvg spending as secondary filter
- **Family Size Impact:** Smaller families (≤2) show highest conversion rates
- **Education Correlation:** Graduate/Advanced degree holders prefer loans (13.0%/13.7% vs 4.4% undergrad)
- **Geographic Patterns:** ZIP codes 92, 94, 93 show higher acceptance rates
- **Digital Engagement:** Online banking users more likely to accept loans

### Code Implementation Examples

**Decision Tree Model Training:**
```python
# Best hyperparameters found through grid search
estimator = DecisionTreeClassifier(
    max_depth=2,
    max_leaf_nodes=50,
    min_samples_split=10,
    class_weight='balanced',
    random_state=42
)
estimator.fit(X_train, y_train)
```

**Cost-Complexity Pruning Implementation:**
```python
# Post-pruning with cost-complexity pruning
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Select best model based on recall optimization
best_ccp_alpha = 0.000272
estimator_2 = DecisionTreeClassifier(
    ccp_alpha=best_ccp_alpha,
    class_weight={0: 0.15, 1: 0.85},
    random_state=1
)
```

**Feature Engineering Pipeline:**
```python
# Data preprocessing and feature engineering
data["Experience"].replace([-1, -2, -3], [1, 2, 3], inplace=True)
data["ZIPCode"] = data["ZIPCode"].astype(str).str[0:2]  # Reduce from 467 to 7 regions
X = pd.get_dummies(data.drop(["Personal_Loan", "Experience"], axis=1),
                  columns=["ZIPCode", "Education"], drop_first=True)
```

**Model Performance Evaluation:**
```python
def model_performance_classification_sklearn(model, predictors, target):
    pred = model.predict(predictors)
    acc = accuracy_score(target, pred)
    recall = recall_score(target, pred)
    precision = precision_score(target, pred)
    f1 = f1_score(target, pred)
    return pd.DataFrame({"Accuracy": acc, "Recall": recall,
                        "Precision": precision, "F1": f1}, index=[0])
```

##  File Structure

```
Personal_Loan_Campaign/
│
├── README.md                    # This file – project overview and documentation
├── Personal_Loan_Campaign.ipynb # Main notebook with code and visualizations
├── data/                        # Folder containing datasets
│   └── personal_loan.csv        # Raw dataset (5,000 customers, 14 features)
├── results/                     # Folder for plots and metrics
│   ├── feature_importance.png   # Feature importance visualization
│   ├── confusion_matrix.png     # Model confusion matrix
│   ├── decision_tree_plot.png   # Decision tree visualization
│   └── pruning_path.png         # Cost-complexity pruning analysis
└── src/                         # Custom functions
    ├── evaluation.py            # model_performance_classification_sklearn(), confusion_matrix_sklearn()
    └── visualization.py         # histogram_boxplot(), labeled_barplot(), distribution_plot_wrt_target()
```

##  Tools & Libraries Used

- **Python Version**: 3.x (tested on 3.11-3.12)
- **Data Processing**: pandas 1.5.3, numpy 1.25.2 for data manipulation
- **Machine Learning**: scikit-learn 1.2.2 for DecisionTreeClassifier, train_test_split, metrics
- **Visualization**: matplotlib 3.7.1, seaborn 0.13.1 for EDA and analysis
- **Development Environment**: Jupyter Notebook / Google Colab with specific version controls
- **Utilities**: warnings module to suppress unnecessary warnings

##  How to Reproduce the Results

1. **Clone the repository:**
```bash
git clone https://github.com/sy22478/Personal_Loan_Campaign.git
cd Personal_Loan_Campaign
```

2. **Install required libraries:**
```bash
pip install pandas==1.5.3 numpy==1.25.2 matplotlib==3.7.1 seaborn==0.13.1 scikit-learn==1.2.2 jupyter
```

3. **Open the Jupyter Notebook:**
```bash
jupyter notebook Personal_Loan_Campaign.ipynb
```
Or upload to Google Colab for cloud execution.

4. **Run all cells sequentially to reproduce:**
   - **Exploratory Data Analysis (EDA)** and visualizations
   - **Data preprocessing** (handling negative values, ZIPCode transformation, one-hot encoding)
   - **Model building** (baseline, pre-pruned, post-pruned decision trees)
   - **Hyperparameter tuning** with GridSearchCV
   - **Cost-complexity pruning** with ccp_alpha optimization
   - **Final evaluation** and model comparison
   - **Business rule extraction** and interpretation

##  Business Recommendations

### Strategic Targeting Priorities:

1. **High-Priority Segments (Immediate Outreach)**:
   - Income > $92.5K + Family ≤ 2
   - Online banking users with high CCAvg spending
   - Graduate/Advanced degree holders in high-income brackets

2. **Medium-Priority Segments (Selective Targeting)**:
   - Income > $92.5K + Family > 2
   - CD Account holders (cross-sell opportunity)
   - ZIP codes 92, 94, 93 (geographic promotions)

3. **Low-Priority Segments (Cost-Benefit Analysis Required)**:
   - Income < $92.5K + CCAvg < $2.95K
   - Undergraduates without online banking
   - High mortgage burden customers

### Operational Improvements:

- **Campaign Efficiency**: Focus on high-recall segments to minimize missed opportunities
- **Cost Reduction**: High precision reduces wasted marketing spend on unlikely converters
- **Personalization**: Use decision rules for tailored messaging (income-based, family-based)
- **Channel Strategy**: Prioritize digital channels for online banking users
- **Regional Focus**: Deploy geographic promotions in high-conversion ZIP codes

##  Final Thoughts

This project successfully built a **classification model** to predict personal loan acceptance using **Decision Trees**. The **post-pruned model** was selected for deployment due to its:
-  **High recall (84.56%)** - minimizes missed opportunities
-  **Good precision (92.65%)** - reduces marketing costs
-  **Interpretability** - actionable business rules for marketing teams
-  **Balance** - optimal trade-off between complexity and performance

### Future Improvements:
1. **Ensemble Methods**: Try Random Forest, Gradient Boosting, or XGBoost for potential performance gains
2. **Threshold Tuning**: Adjust classification threshold to further balance precision and recall
3. **API Integration**: Export the model using joblib/pickle for real-time scoring
4. **Customer Scoring System**: Build a continuous probability score (0-100) for CRM integration
5. **A/B Testing**: Deploy model in production and measure lift vs control group
6. **Feature Engineering**: Create interaction features (Income × CCAvg, Education × Income)
7. **SHAP Values**: Add explainability with SHAP for individual predictions
8. **Model Monitoring**: Implement performance tracking and model drift detection

##  Contact

For questions, improvements, or collaboration:

- **Email**: sonu.yadav19997@gmail.com
- **LinkedIn**: [Sonu Yadav](https://www.linkedin.com/in/sonu-yadav-a61046245/)
- **GitHub**: [@sy22478](https://github.com/sy22478)

---

*This project demonstrates advanced classification techniques with decision trees, handling class imbalance, and translating ML insights into actionable business strategies.*
