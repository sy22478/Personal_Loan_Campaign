# Personal_Loan_Campaign
---

# 🏦 AllLife Bank – Personal Loan Acceptance Prediction

A machine learning classification project to predict whether a liability customer will accept a personal loan based on their demographic and financial attributes.

## 📌 Project Objective

The goal of this project is to build a model that helps **AllLife Bank** identify potential liability customers who are likely to accept personal loans, so they can be targeted more effectively in future campaigns.

This aligns with the bank’s business objective:  
> Convert liability customers (depositors) into asset customers (borrowers), while retaining them as depositors.

---

## 🔍 Key Insights

### 🧠 Top Predictive Features:
| Feature | Importance |
|--------|------------|
| **Income** | Highest |
| **CCAvg** | High |
| **Family Size** | Medium |
| **Education Level** | Moderate |
| **ZIPCode Clusters** | Regional variation noted |

- **Income** was found to be the most important predictor — high-income individuals are significantly more likely to accept personal loans.
- **Credit card usage (CCAvg)** acts as a proxy for financial behavior — higher spenders tend to take more loans.
- **Family size = 3** showed highest likelihood of acceptance.
- Graduate and Advanced degree holders were more likely to accept loans than undergraduates.
- Some ZIP codes (like 92, 94) showed better conversion rates — indicating regional targeting opportunities.

---

## 📊 Dataset Overview

- **Total Rows**: 5000
- **Total Columns**: 14
- **Target Variable**: `Personal_Loan` (binary: 0 = No, 1 = Yes)
- **Class Imbalance**: Only ~9.6% of customers accepted the loan in training data → highly imbalanced dataset

### 📋 Feature List:

| Type | Features |
|------|----------|
| **Numerical** | Age, Income, CCAvg, Family, Mortgage, Experience |
| **Categorical** | Education, ZIPCode, CD_Account, Securities_Account, Online, CreditCard |
| **Dropped** | ID, Experience (due to perfect correlation with Age) |

---

## 🎯 Modeling Approach

Used **Decision Tree Classifier** due to its interpretability and ability to handle class imbalance via `class_weight='balanced'`.

### 🧪 Baseline Model Performance (Training):
| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| Recall | 100% |
| Precision | 100% |
| F1-score | 100% |

> ⚠️ Overfitting observed due to deep tree structure

---

### 🔧 Pre-Pruned Decision Tree (Hyperparameter Tuned)

```python
max_depth=2, max_leaf_nodes=50, min_samples_split=10
```

| Metric | Training | Test |
|--------|----------|------|
| Accuracy | 79.0% | 77.9% |
| Recall | 100% | 100% |
| Precision | 31.1% | 31.0% |
| F1-score | 47.4% | 47.4% |

✅ **Perfect recall achieved** – no missed opportunities  
❗ Lower precision suggests many false positives – acceptable if outreach cost is low

---

### 🌳 Post-Pruned Decision Tree (Cost-Complexity Pruning)

Selected best model using `ccp_alpha` values from pruning path.

Final performance:
| Metric | Training | Test |
|--------|----------|------|
| Accuracy | 99.97% | 97.8% |
| Recall | 100% | 84.56% |
| Precision | 99.70% | 92.65% |
| F1-score | 99.85% | 88.42% |

✅ Achieves a good balance between recall and precision  
🧾 Simple decision rules can be shared with marketing teams

---

## 📈 Best Performing Model

**Post-pruned Decision Tree** was selected as the final model because:
- It balances complexity and generalization
- Maintains high recall (minimizes missed opportunities)
- Has better precision than pre-pruned models
- Is interpretable by stakeholders

---

## 📋 Final Decision Rules (Simplified)

From the pruned tree:
```
|--- Income <= 92.5
|    |--- CCAvg <= 2.95 → Definitely no
|    |--- CCAvg > 2.95 → Maybe yes
|--- Income > 92.5
     |--- Family <= 2 → Definitely yes
     |--- Family > 2 → Probably yes
```

These rules can guide manual targeting and CRM segmentation.

---

## 🧩 File Structure

```
Module2_PersonalLoan_Prediction/
│
├── README.md                    # This file – project overview and structure
├── Module2_Project.ipynb         # Main notebook with code and visualizations
├── data/                        # Folder containing datasets
│   └── personal_loan.csv        # Raw dataset used in the project
├── results/                     # Folder for plots and metrics
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── decision_tree_plot.png
└── src/                         # Custom functions
    ├── evaluation.py            # Functions: model_performance_classification_sklearn(), confusion_matrix_sklearn()
    └── visualization.py         # Functions: histogram_boxplot(), labeled_barplot(), distribution_plot_wrt_target()
```

---

## 🛠 Tools & Libraries Used

- **Python Version**: 3.x
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation
  - `matplotlib`, `seaborn` – Visualizations
  - `sklearn` – DecisionTreeClassifier, train_test_split, metrics
  - `warnings` – To suppress unnecessary warnings

---

## 📁 How to Reproduce the Results

1. Clone the repo:
```bash
git clone https://github.com/yourname/Module2_PersonalLoan_Prediction.git
```

2. Install required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open the Jupyter Notebook or Google Colab file:
```bash
jupyter notebook Module2_Project.ipynb
```

4. Run all cells sequentially to reproduce:
   - EDA and visualizations
   - Data preprocessing
   - Model building and tuning
   - Final evaluation and comparison

---

## 📈 Business Recommendations

| Segment | Likelihood to Accept Loan | Recommended Strategy |
|--------|---------------------------|---------------------|
| **Income > $92.5K + Family ≤ 2** | Very High | Top priority for outreach |
| **Income > $92.5K + Family > 2** | High | Include in campaigns |
| **Income < $92.5K + CCAvg > $2.95K** | Medium-High | Target selectively |
| **Income < $92.5K + CCAvg < $2.95K** | Low | Avoid unless cost-effective |
| **CD Account Holder** | Medium | Cross-sell via relationship managers |
| **Graduate Degree** | Medium | Use targeted content |
| **ZIPCode 92/94/93** | Medium-High | Geographic promotions |

---

## 📝 Final Thoughts

This project successfully built a classification model to predict personal loan acceptance using Decision Trees. The post-pruned model was selected for deployment due to its **high recall**, **good precision**, and **interpretability**.

Future improvements could include:
- Trying **ensemble methods** like Random Forest or XGBoost
- Using **threshold tuning** to further balance precision and recall
- Exporting the model for **API integration**
- Building a **customer scoring system** for real-time targeting

---

## 📬 Contact

For questions or improvements, feel free to reach out:

- **Email**: sonu.yadav19997@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/sonu-yadav-a61046245/
