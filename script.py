import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# 1) Load
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
X = df.drop(columns=['target'])
y = df['target']  # 0=malignant, 1=benign (sklearn encoding)

# 2) VIF filter (remove multicollinearity)
def calculate_vif(df):
    vif = pd.DataFrame()
    vif["feature"] = df.columns
    vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif

X_vif = X.copy()
while True:
    vif_df = calculate_vif(X_vif)
    high = vif_df.sort_values("VIF", ascending=False).iloc[0]
    if high.VIF > 10:
        X_vif = X_vif.drop(columns=[high.feature])
    else:
        break

# 3) Backward elimination by p-value (statsmodels)
X_sel = sm.add_constant(X_vif)
while True:
    logit = sm.Logit(y, X_sel).fit(disp=False)
    pvals = logit.pvalues
    worst = pvals.idxmax()
    if pvals[worst] > 0.05 and worst != 'const':
        X_sel = X_sel.drop(columns=[worst])
    else:
        break

selected_features = [c for c in X_sel.columns if c != 'const']

# 4) Train/test split + scale
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features], y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 5) Fit sklearn Logistic Regression
clf = LogisticRegression(solver="liblinear", penalty="l2", max_iter=1000)
clf.fit(X_train, y_train)

# 6) Confusion matrix (Benign/Malignant labels)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[1,0])  # rows: true, cols: pred
# Reorder to show rows/cols as: Benign, Malignant
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted label"); plt.ylabel("True label")
plt.show()

# 7) Metrics
print(classification_report(y_test, y_pred, target_names=['Malignant','Benign']))

# 8) ROC–AUC
y_proba = clf.predict_proba(X_test)[:,1]  # probability of Benign(=1)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.show()
