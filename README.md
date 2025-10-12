# 🏦 AllLife Bank – Personal Loan Campaign Analysis

## 📌 Project Overview
A machine learning classification project for **AllLife Bank** to predict whether liability customers will accept personal loan offers based on their demographic and financial attributes. Uses Decision Tree algorithms with pre-pruning and post-pruning techniques to optimize customer targeting and campaign effectiveness while handling class imbalance in the dataset.

### Business Objective
> Convert liability customers (depositors) into asset customers (borrowers), while retaining them as depositors.

This project helps the bank identify potential liability customers who are likely to accept personal loans, so they can be targeted more effectively in future campaigns, improving the previous campaign's **9% success rate**.

## 🔍 Key Insights

### 🧠 Top Predictive Features:
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
│   Profiles      │───►│  Preprocessing   │───►│   Modeling      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                ┌──────▼──────┐                │
         │                │ Class       │                │
         │                │ Imbalance   │                │
         │                │ Handling    │                │
         │                └──────┬──────┘                │
         │                       │                       │
         │                ┌──────▼──────┐                │
         │                │ Pre-/Post   │                │
         │                │ Pruning     │                │
         │                └─────────────┘                │
         │                                               ▼
┌─────────────────┐                               ┌───────────────┐
│   Evaluation    │◄──────────────────────────────│  Deployment   │
│  (Recall/ROC)   │                               │  (Optional)   │
└─────────────────┘                               └───────────────┘
```

## 🧪 Methodology

### Data Preprocessing
- Handle class imbalance using stratified sampling or class weights
- Encode categorical variables (One-Hot/Ordinal)
- Feature scaling for continuous variables where applicable

### Modeling
- Decision Tree classifier with GridSearchCV for pre-pruning hyperparameters (max_depth, min_samples_split, min_samples_leaf)
- Post-pruning using cost complexity pruning (ccp_alpha)
- Train/validation/test splits for robust evaluation

### Evaluation Metrics
- ROC-AUC, Recall, Precision, F1-score
- Confusion Matrix to monitor False Negatives (missed acceptors)

## 📈 Results Summary
- Income > $92.5K and CCAvg > $2.95K produce a high-lift segment
- Graduate and Advanced education segments show 3x higher acceptance than undergrad
- Smaller families (≤2) show higher acceptance
- Certain ZIP code clusters show above-average conversion rates

## 📁 Project Structure
```
Personal_Loan_Campaign/
├── data/
│   ├── bank_train.csv
│   └── bank_test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling_prepruning.ipynb
│   └── 03_post_pruning_evaluation.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train model with grid search
python -m src.model --grid

# Evaluate on test set
python -m src.evaluate
```

## 🔮 Recommendations
- Use targeted campaigns for income >$92.5K and high CCAvg users
- Prioritize graduates/advanced degree holders in outreach
- Focus on ZIP codes 92, 94, 93 for regional marketing
- Increase digital marketing for online banking users

## 👤 Author
- GitHub: @sy22478
- LinkedIn: https://www.linkedin.com/in/sonu-yadav-a61046245/
