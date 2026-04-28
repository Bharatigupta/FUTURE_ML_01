# ============================================================
# FUTURE_ML_01 — Sales & Demand Forecasting System
# Future Interns | Machine Learning Track | Task 1
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  FUTURE_ML_01 — Sales & Demand Forecasting System")
print("  Future Interns | ML Track | Task 1")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────
print("\n[1/7] Loading dataset...")
df = pd.read_csv('../data/sales_data.csv')
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
print(f"  ✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  ✓ Date range: {df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")

# ── 2. DATA CLEANING & EDA ─────────────────────────────────
print("\n[2/7] Data Cleaning & EDA...")
print(f"  ✓ Missing values: {df.isnull().sum().sum()}")
print(f"  ✓ Total Sales: ${df['Sales'].sum():,.2f}")
print(f"  ✓ Total Profit: ${df['Profit'].sum():,.2f}")
print(f"  ✓ Categories: {df['Category'].unique()}")

# Monthly aggregation
df['Year'] = df['Order_Date'].dt.year
df['Month'] = df['Order_Date'].dt.month
df['Quarter'] = df['Order_Date'].dt.quarter
df['YearMonth'] = df['Order_Date'].dt.to_period('M')

monthly_sales = df.groupby('YearMonth')['Sales'].sum().reset_index()
monthly_sales['YearMonth_str'] = monthly_sales['YearMonth'].astype(str)

# ── 3. VISUALIZATIONS ─────────────────────────────────────
print("\n[3/7] Generating EDA Visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Sales & Demand — Exploratory Data Analysis', fontsize=16, fontweight='bold', y=1.01)

# Plot 1: Monthly Sales Trend
ax1 = axes[0, 0]
ax1.plot(range(len(monthly_sales)), monthly_sales['Sales'], color='#2196F3', linewidth=2, marker='o', markersize=4)
ax1.fill_between(range(len(monthly_sales)), monthly_sales['Sales'], alpha=0.15, color='#2196F3')
step = max(1, len(monthly_sales) // 8)
ax1.set_xticks(range(0, len(monthly_sales), step))
ax1.set_xticklabels(monthly_sales['YearMonth_str'].iloc[::step], rotation=45, ha='right', fontsize=8)
ax1.set_title('Monthly Sales Trend', fontweight='bold')
ax1.set_xlabel('Month')
ax1.set_ylabel('Total Sales ($)')
ax1.grid(True, alpha=0.3)

# Plot 2: Sales by Category
cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
colors = ['#FF5722', '#4CAF50', '#9C27B0']
bars = ax1_2 = axes[0, 1]
bars.bar(cat_sales.index, cat_sales.values, color=colors, edgecolor='white', linewidth=1.5)
for i, (cat, val) in enumerate(zip(cat_sales.index, cat_sales.values)):
    bars.text(i, val + 500, f'${val:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
bars.set_title('Sales by Category', fontweight='bold')
bars.set_ylabel('Total Sales ($)')
bars.set_xlabel('Category')
bars.grid(True, alpha=0.3, axis='y')

# Plot 3: Regional Sales
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
axes[1, 0].pie(region_sales.values, labels=region_sales.index,
               autopct='%1.1f%%', colors=['#2196F3','#FF5722','#4CAF50','#FFC107'],
               startangle=90, wedgeprops={'edgecolor':'white', 'linewidth':2})
axes[1, 0].set_title('Sales Distribution by Region', fontweight='bold')

# Plot 4: Monthly Sales by Category (heatmap style)
pivot = df.groupby(['Year', 'Month'])['Sales'].sum().unstack(fill_value=0)
sns.heatmap(pivot, ax=axes[1, 1], cmap='YlOrRd', annot=True, fmt='.0f',
            linewidths=0.5, cbar_kws={'label': 'Sales ($)'})
axes[1, 1].set_title('Sales Heatmap (Year × Month)', fontweight='bold')
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Year')

plt.tight_layout()
plt.savefig('../data/eda_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ EDA charts saved → data/eda_distribution.png")

# ── 4. FEATURE ENGINEERING ────────────────────────────────
print("\n[4/7] Feature Engineering...")
le = LabelEncoder()
df_ml = df.copy()
for col in ['Category', 'Sub_Category', 'Region', 'Segment', 'State']:
    df_ml[col + '_enc'] = le.fit_transform(df_ml[col])

# Lag features (monthly)
monthly_ml = df_ml.groupby(['Year', 'Month']).agg(
    Total_Sales=('Sales', 'sum'),
    Total_Profit=('Profit', 'sum'),
    Avg_Discount=('Discount', 'mean'),
    Total_Quantity=('Quantity', 'sum'),
    Num_Orders=('Order_ID', 'count')
).reset_index()

monthly_ml['Lag_1'] = monthly_ml['Total_Sales'].shift(1)
monthly_ml['Lag_2'] = monthly_ml['Total_Sales'].shift(2)
monthly_ml['Lag_3'] = monthly_ml['Total_Sales'].shift(3)
monthly_ml['Rolling_Mean_3'] = monthly_ml['Total_Sales'].rolling(3).mean()
monthly_ml = monthly_ml.dropna()

features = ['Year', 'Month', 'Lag_1', 'Lag_2', 'Lag_3',
            'Rolling_Mean_3', 'Avg_Discount', 'Total_Quantity', 'Num_Orders']
X = monthly_ml[features]
y = monthly_ml['Total_Sales']

print(f"  ✓ Features created: {features}")
print(f"  ✓ Training samples: {len(X)}")

# ── 5. MODEL TRAINING ─────────────────────────────────────
print("\n[5/7] Training Models...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {name}:")
    print(f"    MAE  = ${mae:,.2f}")
    print(f"    RMSE = ${rmse:,.2f}")
    print(f"    R²   = {r2:.4f}")
    return mae, rmse, r2

print("\n  📊 Model Evaluation Results:")
print("  " + "-" * 45)
mae_lr, rmse_lr, r2_lr = evaluate("Linear Regression", y_test, y_pred_lr)
mae_rf, rmse_rf, r2_rf = evaluate("Random Forest    ", y_test, y_pred_rf)

best_model = rf if r2_rf > r2_lr else lr
best_name = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
print(f"\n  🏆 Best Model: {best_name}")

# ── 6. VISUALIZATIONS — FORECAST ──────────────────────────
print("\n[6/7] Generating Forecast Visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Sales Forecast — Model Comparison', fontsize=15, fontweight='bold')

# Plot 1: Actual vs Predicted
x_idx = range(len(y_test))
axes[0].plot(x_idx, y_test.values, 'o-', color='#2196F3', label='Actual Sales', linewidth=2, markersize=6)
axes[0].plot(x_idx, y_pred_lr, 's--', color='#FF5722', label='Linear Regression', linewidth=1.5, markersize=5)
axes[0].plot(x_idx, y_pred_rf, '^--', color='#4CAF50', label='Random Forest', linewidth=1.5, markersize=5)
axes[0].set_title('Actual vs Predicted Sales', fontweight='bold')
axes[0].set_xlabel('Test Period (Months)')
axes[0].set_ylabel('Sales ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Model Comparison Bar Chart
models = ['Linear\nRegression', 'Random\nForest']
r2_scores = [r2_lr, r2_rf]
mae_scores = [mae_lr, mae_rf]
bar_colors = ['#FF5722', '#4CAF50']
bars = axes[1].bar(models, r2_scores, color=bar_colors, edgecolor='white', linewidth=2, width=0.5)
for bar, score in zip(bars, r2_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'R² = {score:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
axes[1].set_title('Model Comparison (R² Score)', fontweight='bold')
axes[1].set_ylabel('R² Score (Higher = Better)')
axes[1].set_ylim(0, 1.1)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
axes[1].legend()

plt.tight_layout()
plt.savefig('../data/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Model comparison chart saved → data/model_comparison.png")

# Feature Importance Plot
fig, ax = plt.subplots(figsize=(10, 6))
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
colors_fi = ['#FF5722' if v == feat_imp.max() else '#2196F3' for v in feat_imp.values]
feat_imp.plot(kind='barh', ax=ax, color=colors_fi, edgecolor='white')
ax.set_title('Feature Importance — Random Forest', fontweight='bold', fontsize=13)
ax.set_xlabel('Importance Score')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../data/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Feature importance chart saved → data/feature_importance.png")

# ── 7. SAVE MODELS ────────────────────────────────────────
print("\n[7/7] Saving Models...")
with open('../model/linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
with open('../model/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("  ✓ linear_regression_model.pkl saved")
print("  ✓ random_forest_model.pkl saved")

# ── FUTURE FORECAST (Next 3 Months) ───────────────────────
print("\n  📅 Future Forecast (Next 3 Months):")
print("  " + "-" * 40)
last_row = monthly_ml.iloc[-1]
last_sales = last_row['Total_Sales']
last_month = int(last_row['Month'])
last_year = int(last_row['Year'])

for i in range(1, 4):
    next_month = (last_month + i - 1) % 12 + 1
    next_year = last_year + ((last_month + i - 1) // 12)
    future_feat = np.array([[next_year, next_month,
                              last_sales, last_sales * 0.98, last_sales * 0.96,
                              last_sales * 0.98, 0.15, 400, 120]])
    pred = best_model.predict(future_feat)[0]
    print(f"  {next_year}-{next_month:02d}: Predicted Sales = ${pred:,.2f}")

print("\n" + "=" * 60)
print("  ✅ Task 1 Complete! All outputs saved.")
print("=" * 60)
