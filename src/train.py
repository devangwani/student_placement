import json, joblib, re
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, classification_report

RANDOM_STATE = 42

# --- Utilities ---
def to_bool01(x):
    """Map common yes/no-like strings to 1/0, otherwise return original."""
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"yes","y","true","t","1"}: return 1
        if s in {"no","n","false","f","0"}: return 0
    return x

def to_numeric_clean(s):
    """Remove %, commas, spaces; return numeric or NaN."""
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float, np.number)): return s
    s = str(s).strip()
    s = s.replace(",", "")
    s = s.replace("%", "")
    s = re.sub(r"\s+", "", s)
    try:
        return float(s)
    except:
        return np.nan

def tune_threshold(y_true, y_proba):
    p, r, t = precision_recall_curve(y_true, y_proba)
    f1 = 2*(p*r)/(p+r+1e-9)
    idx = np.nanargmax(f1)
    thr = float(t[max(idx-1, 0)]) if len(t) > 0 else 0.5
    return thr, float(f1[idx])

def load_config():
    with open("config.json") as f:
        return json.load(f)

def map_target_series(y):
    if y.dtype == object:
        return y.astype(str).str.lower().str.strip().isin(["placed","1","yes","true"]).astype(int)
    return y.astype(int)

# --- Main ---
def main():
    cfg = load_config()
    features_found = [c for c in cfg["features_found"] if c is not None]
    if not features_found:
        raise RuntimeError("No usable feature columns detected. Edit config.json or this script.")

    target = cfg["target_col"]
    df = pd.read_csv(cfg["raw_csv_path"])

    # Clean feature columns: boolean-like -> 1/0, numeric-like strings -> float
    X = df[features_found].copy()
    # Force convert SSC/HSC if present (e.g., "85%")
    for col in ["SSC_Marks", "HSC_Marks"]:
        if col in X.columns:
            X[col] = (X[col]
                  .astype(str)
                  .str.replace(",", "", regex=False)
                  .str.replace("%", "", regex=False)
                  .str.strip())
            X[col] = pd.to_numeric(X[col], errors="coerce")


    # First pass: map common boolean strings to 1/0
    for c in X.columns:
        X[c] = X[c].map(to_bool01)

    # Second pass: try to coerce to numeric where possible (without breaking genuine categories)
    # Heuristic: if column is object and majority values look numeric/percent-like, coerce entire col
    for c in X.columns:
        if X[c].dtype == object:
            sample = X[c].dropna().astype(str).head(50).tolist()
            if len(sample) > 0:
                looks_num = sum(bool(re.fullmatch(r"[+-]?\d+(\.\d+)?%?", s.replace(",", ""))) for s in sample)
                if looks_num / len(sample) >= 0.7:
                    X[c] = X[c].map(to_numeric_clean)

    # Target
    y = map_target_series(df[target])

    # Split numeric vs categorical
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Pipelines
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ],
        remainder="drop"
    )

    clf = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    pipe = Pipeline([("preprocess", pre), ("clf", clf)])

    # Train/validation split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # CV predictions on train for threshold selection
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    yproba_cv = cross_val_predict(pipe, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    thr, best_f1 = tune_threshold(ytr.values, yproba_cv)

    # Fit on full train
    pipe.fit(Xtr, ytr)

    # Evaluate on test
    yproba_te = pipe.predict_proba(Xte)[:, 1]
    ypred_te = (yproba_te >= thr).astype(int)
    f1 = f1_score(yte, ypred_te)
    auprc = average_precision_score(yte, yproba_te)

    print({"threshold": thr, "test_f1": f1, "test_auprc": auprc, "cv_best_f1": best_f1})
    print("\nClassification report (test):\n", classification_report(yte, ypred_te, digits=3))
    print(f"\nDetected numeric cols: {num_cols}")
    print(f"Detected categorical cols: {cat_cols}")

    # Save artifacts
    # We save the fitted preprocessor and classifier separately, for the Streamlit app
    joblib.dump(pipe.named_steps["preprocess"], "models/preprocessor.joblib")
    joblib.dump(pipe.named_steps["clf"], "models/classifier.joblib")
    with open("models/threshold.json", "w") as f:
        json.dump({"threshold": thr}, f)

    # Save placed cohort medians (based on original feature columns; missing ones ignored)
    try:
        placed_df = df[features_found][df[target].astype(str).str.lower().str.strip().isin(["placed","1","yes","true"])]
        # Compute medians only on numeric interpretations
        placed_num = placed_df.copy()
        for c in placed_num.columns:
            placed_num[c] = placed_num[c].map(to_bool01)
            placed_num[c] = placed_num[c].map(to_numeric_clean) if placed_num[c].dtype == object else placed_num[c]
        med = {c: float(pd.to_numeric(placed_num[c], errors="coerce").median()) for c in placed_num.columns}
    except Exception:
        med = {}
    with open("models/cohort_medians.json", "w") as f:
        json.dump(med, f)

if __name__ == "__main__":
    main()
