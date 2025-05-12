#main2
# if not installed, I am using the OS libraries pandas, numpy, scikit-learn, seaborn, matplotlib, statsmodels (pip install + name in the terminal)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
import seaborn as sns
# ---------- Loading Data ----------
df = pd.read_csv(r"C:\Users\bruno\Downloads\loan_dataset.csv")
# ---------- Loading (translating to binary) and defining outcome vars + regressors ----------
df["loan_status_binary"] = df["loan_status"].map({"repaid": 0, "default": 1})
y = df["loan_status_binary"]
X = df.drop(columns=["customer_id", "application_date", "loan_status", "loan_status_binary","zip_code_prefix"])

# ---------- Preprocessing ----------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
])
# I realize that preprocessing was meant to be already done, but as I am using libraries I wanted to make sure everything was set up correctly
# The OneHotEncoder() converts some variable into binary "indicator columns", dummy vars like:"education_phd" and "education_masters".
#it also drops one observation to avoid possible multicollinearity issues

# ---------- Logistic Regression ----------
clf = Pipeline([
    ("pre", preprocessor),
    ("logit", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# ---------- Predicting Default Probability ----------
p_default = clf.predict_proba(X_test)[:, 1]

# ---------- Base Case: Approval if Expected Profit > 0 ----------
R = 1
L = 1
pi_s = (1 - p_default) * R + p_default * (-L)
y_pred_profit = (pi_s > 0).astype(int)
y_pred_default = 1 - y_pred_profit  # align with default = 1

# testing the accuracy with a of BOTH predicted(default)=actual(default) and predicted(repaid)=actual(repaid)
accuracy = accuracy_score(y_test, y_pred_default)
print(f"Accuracy with expected profit rule: {accuracy:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_default, target_names=["Repaid", "Default"]))

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred_default)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Repaid", "Default"])
disp.plot()
plt.show()

# -----------guessing R & L from the data, yields worse predicting power ------------

R = 0.2 * df["loan_amount"].mean()
L = df["loan_amount"].mean()
pi_s = (1 - p_default) * R + p_default * (-L)
y_pred_profit = (pi_s > 0).astype(int)
y_pred_default = 1 - y_pred_profit  # align with default = 1

accuracy = accuracy_score(y_test, y_pred_default)
print(f"Accuracy with expected profit rule: {accuracy:.4f}")
# -----------the same applies to other combinations where they are different from each other ------------

# ------------------checking if there is multicollinearity on some regressors-----------------
# Extracting the preprocessed design matrix 
X_processed = preprocessor.fit_transform(X)
# Converting to a DataFrame with named columns and
# Getting column names from the transformer
encoded_cat_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + encoded_cat_cols.tolist()
X_df = pd.DataFrame(X_processed, columns=all_feature_names)

#Computing VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

print("\n🔍 Variance Inflation Factors:")
print(vif_data.sort_values("VIF", ascending=False))
# values >5 tend be quite collinear, and a few are displaying high values, running a corr matrix
# Select top-VIF variables
cols = ["payment_to_income_ratio", "loan_to_income_ratio", "monthly_payment", "loan_amount", "income"]
# Plot correlation matrix
corr_matrix = df[cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Between High-VIF Variables")
plt.show()
# after dropping payment to income ratio for redundancy (you can modify the exclusion above at line 20 to do so)
# the accuracy improves by .01%, hence this modification might only be useful when interpreting the single 
#coefficient(s) and does not impact accuracy so much, noting in the report.

# Keeping the model as is for the now, printing here the top predictors 
# as it's not ols these betas are to be thought as increases in log-odds probability, also, because I modeled the p of default the repayment coeffs
# are rightfully negative
coef = clf.named_steps["logit"].coef_[0]
features = preprocessor.get_feature_names_out()
top_features = pd.Series(coef, index=features).sort_values(ascending=False)
print(top_features.head(5))  # top predictors of default
print(top_features.tail(5))  # top predictors of repayment

#-------- relatively redundant: checking for McFadden's R^2 (or likelihood ratio index) with statsmodels -------------------
# Add constant (intercept)
X_sm = sm.add_constant(X_df)

#  fittingg the logistic regression model and extracting loglikelihoods
logit_model = sm.Logit(y, X_sm).fit()
ll_full = logit_model.llf                  # Log-likelihood of full model
ll_null = logit_model.llnull               # Log-likelihood of null model
k = logit_model.df_model + 1               # Number of parameters (including intercept)

# McFadden R^2
r2_mcfadden = 1 - (ll_full / ll_null)

# Adjusted McFadden R^2
r2_adj_mcfadden = 1 - ((ll_full - k) / ll_null)

print(f"McFadden R²: {r2_mcfadden:.4f}")
print(f"Adjusted McFadden R²: {r2_adj_mcfadden:.4f}")