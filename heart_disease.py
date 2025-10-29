# %% [markdown]
# Heart Disease — Classification

# %%
import pandas as pd
import matplotlib.pyplot as plt
import ucimlrepo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# %% [markdown]
# Load dataset and show basic shape
#
# %%
ds = ucimlrepo.fetch_ucirepo(id=45)

print(f"Loaded heart dataset with shape: {ds.shape}")


# %% [markdown]
# Basic preprocessing and train/test split
#
# %%
# Basic preprocessing: user may need to update target column name
X = ds.data.features
y = ds.data.targets

missing_cols = X.columns[X.isnull().any()].tolist()
print("Columns with missing values:", missing_cols)

# If there are missing values, we can fill them with median
if missing_cols:
    for col in missing_cols:
        X.fillna({ col: X[col].median() }, inplace=True)
        
# y_binary = (y['num'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# %% [markdown]
# Define model pipelines (scaling + estimator)
#
# %%
mlp_pipe = Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42))])
svm_pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=42))])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# %% [markdown]
# Evaluation helper: cross-validated train predictions and final test eval
#
# %%
def evaluate(pipe, X_tr, y_tr, X_te, y_te, cv):
	y_pred_cv = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict")
	p, r, f1, _ = precision_recall_fscore_support(y_tr, y_pred_cv, average="binary")
	acc_cv = accuracy_score(y_tr, y_pred_cv)

	pipe.fit(X_tr, y_tr)
	y_pred_test = pipe.predict(X_te)
	p_t, r_t, f1_t, _ = precision_recall_fscore_support(y_te, y_pred_test, average="binary")
	acc_t = accuracy_score(y_te, y_pred_test)

	return {"cv": {"precision": p, "recall": r, "f1": f1, "accuracy": acc_cv}, "test": {"precision": p_t, "recall": r_t, "f1": f1_t, "accuracy": acc_t}, "model": pipe, "y_pred_test": y_pred_test}


# %%
res_mlp = evaluate(mlp_pipe, X_train, y_train, X_test, y_test, cv)
res_svm = evaluate(svm_pipe, X_train, y_train, X_test, y_test, cv)

print("MLP test accuracy:", res_mlp["test"]["accuracy"])
print("SVM test accuracy:", res_svm["test"]["accuracy"])


# %% [markdown]
# PCA experiment (95% variance)
#
# %%
pca95 = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95))])
X_train_95 = pca95.fit_transform(X_train)
X_test_95 = pca95.transform(X_test)

mlp_95 = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
svm_95 = SVC(kernel="rbf", probability=True, random_state=42)

mlp_95.fit(X_train_95, y_train)
svm_95.fit(X_train_95, y_train)

print("MLP PCA95 acc:", accuracy_score(y_test, mlp_95.predict(X_test_95)))
print("SVM PCA95 acc:", accuracy_score(y_test, svm_95.predict(X_test_95)))


# %% [markdown]
# Visualizations: simple comparison of test metrics
#
# %%
metrics_df = pd.DataFrame([
	{"model": "MLP", **res_mlp["test"]},
	{"model": "SVM", **res_svm["test"]},
])
metrics_df_plot = metrics_df.set_index("model").T
metrics_df_plot.plot.bar(rot=0)
plt.title("Heart disease: test metrics comparison")
plt.tight_layout()
plt.show()

