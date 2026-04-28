# 📈 FUTURE_ML_01 — Sales & Demand Forecasting System

**Future Interns | Machine Learning Track | Task 1**

---

## 📌 Project Overview

An ML-based Sales & Demand Forecasting system that uses historical business data to predict future sales trends — helping businesses plan inventory, optimize budgets, and make smarter decisions.

---

## 🎯 What It Does

- **Forecasts** monthly sales using historical time-series data
- **Compares** two ML models: Linear Regression vs Random Forest
- **Identifies** key demand drivers using feature importance analysis
- **Predicts** future sales for next N months
- **Visualizes** trends, regional patterns, and model performance

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | ML models and evaluation |
| Matplotlib / Seaborn | Visualizations |
| Pickle | Saving trained models |

---

## 🤖 ML Pipeline

```
Raw Sales Data
      ↓
Data Cleaning & EDA (missing values, outliers, distributions)
      ↓
Time-Based Feature Engineering
(Year, Month, Lag_1, Lag_2, Lag_3, Rolling_Mean_3)
      ↓
Train/Test Split (80% / 20%)
      ↓
┌──────────────────┐     ┌─────────────────────┐
│ Linear Regression│     │   Random Forest      │
└──────────────────┘     └─────────────────────┘
      ↓                          ↓
      ─────────── Evaluation ────────────
                    ↓
           Best Model → Saved (.pkl)
                    ↓
     Forecast: Next 3 Months of Sales
```

---

## 📊 Results

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | ~$0 | ~$0 | ~1.00 |
| Random Forest | ~$5,073 | ~$7,759 | ~0.26 |

> 🏆 **Best Model: Linear Regression** achieved near-perfect accuracy on monthly aggregated data

---

## 💡 Key Learnings

- **Lag features** are the most powerful predictors in time-series forecasting
- **Rolling mean** smooths out noise and improves model stability
- **Linear Regression** outperforms Random Forest on structured time-series data
- **Seasonal patterns** (Q4 boost) are clearly visible in monthly trends
- **Feature importance** reveals which variables drive demand the most

---

## 📁 Project Structure

```
FUTURE_ML_01/
├── data/
│   ├── sales_data.csv              ← Dataset
│   ├── generate_data.py            ← Data generation script
│   ├── eda_distribution.png        ← EDA charts
│   ├── model_comparison.png        ← Model comparison plot
│   └── feature_importance.png      ← Feature importance chart
├── model/
│   ├── linear_regression_model.pkl ← Saved Linear Regression model
│   └── random_forest_model.pkl     ← Saved Random Forest model
├── notebook/
│   └── sales_forecasting.py        ← Main project code
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Bharatigupta/FUTURE_ML_01.git
cd FUTURE_ML_01

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
cd data
python generate_data.py

# 4. Run the forecasting model
cd ../notebook
python sales_forecasting.py
```

---

## 📈 Sample Forecast Output

```
📅 Future Forecast (Next 3 Months):
  2024-01: Predicted Sales = $43,599.58
  2024-02: Predicted Sales = $43,599.58
  2024-03: Predicted Sales = $43,599.58
```

---

## 👤 Author

**Bharat Gupta** — Future Interns | ML Track
GitHub: [@Bharatigupta](https://github.com/Bharatigupta)
