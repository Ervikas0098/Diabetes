"""
DiabetesSense v2.0 — ML Training Pipeline
==========================================
Dataset:  Early Stage Diabetes Risk Prediction (UCI)
Source:   https://archive.ics.uci.edu/dataset/529
Features: Age, Gender + 14 binary symptom indicators
Target:   class (Positive / Negative)

Run:
    python models/train.py

Outputs:
    models/best_model.pkl     — Trained Random Forest model
    models/scaler.pkl         — StandardScaler (for LR/SVM/MLP if needed)
    models/model_meta.json    — Metadata, results, feature importances
    static/model_evaluation.png
    static/confusion_matrix.png
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATA_PATH   = "data/diabetes_data_upload.csv"
MODEL_DIR   = "models"
STATIC_DIR  = "static"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# Binary feature encoding maps
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
TARGET_MAP = {"Positive": 1, "Negative": 0}

# Models that require feature scaling
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM", "Neural Network (MLP)"}


# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, remove duplicates, and encode all categorical variables."""
    print(f"\n{'='*60}")
    print("STEP 1 — DATA LOADING & CLEANING")
    print(f"{'='*60}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from: https://archive.ics.uci.edu/dataset/529\n"
            "Save as: data/diabetes_data_upload.csv"
        )

    df = pd.read_csv(path)
    print(f"  ✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Features: {list(df.columns[:-1])}")
    print(f"  Target:   '{df.columns[-1]}'")
    print(f"  Class:    {dict(df['class'].value_counts())}")
    print(f"  Dtypes:   {dict(df.dtypes.value_counts())}")
    print(f"  Missing:  {df.isnull().sum().sum()} total")
    print(f"  Dups:     {df.duplicated().sum()} → removing...")

    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  ✓ After dedup: {df.shape[0]} rows")

    return df


# ─── 2. ENCODE ───────────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame):
    """Encode categorical variables to numeric, return X, y, feature names."""
    print(f"\n{'='*60}")
    print("STEP 2 — FEATURE ENCODING")
    print(f"{'='*60}")

    df_enc = pd.DataFrame()
    df_enc["Age"] = df["Age"].astype(int)

    # Encode Gender and all binary symptom columns
    for col in df.columns:
        if col == "Age":
            continue
        elif col == "class":
            df_enc["class"] = df[col].map(TARGET_MAP)
        else:
            df_enc[col] = df[col].map(BINARY_MAP)

    # Verify no nulls after encoding
    null_count = df_enc.isnull().sum().sum()
    if null_count > 0:
        print(f"  ⚠ {null_count} nulls after encoding — filling with mode...")
        df_enc = df_enc.fillna(df_enc.mode().iloc[0])

    feature_cols = [c for c in df_enc.columns if c != "class"]
    X = df_enc[feature_cols].values.astype(float)
    y = df_enc["class"].values.astype(int)

    print(f"  ✓ Features ({len(feature_cols)}): {feature_cols}")
    print(f"  ✓ Class balance: {y.sum()} positive ({y.mean()*100:.1f}%) | "
          f"{(1-y).sum()} negative ({(1-y).mean()*100:.1f}%)")

    return X, y, feature_cols


# ─── 3. SPLIT + SCALE ────────────────────────────────────────────────────────
def prepare_data(X, y):
    """Stratified split and StandardScaler fit."""
    print(f"\n{'='*60}")
    print("STEP 3 — TRAIN/TEST SPLIT")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  ✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"  ✓ Stratified — Train pos: {y_train.mean()*100:.1f}% | "
          f"Test pos: {y_test.mean()*100:.1f}%")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler


# ─── 4. DEFINE MODELS ────────────────────────────────────────────────────────
def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, random_state=RANDOM_SEED
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, weights="distance", n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=RANDOM_SEED
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            alpha=0.001, learning_rate="adaptive",
            random_state=RANDOM_SEED
        ),
    }


# ─── 5. TRAIN & EVALUATE ─────────────────────────────────────────────────────
def evaluate_all(models, X_train, X_test, y_train, y_test,
                 X_train_sc, X_test_sc) -> tuple[list, dict]:
    print(f"\n{'='*60}")
    print("STEP 4 — MODEL TRAINING & EVALUATION")
    print(f"{'='*60}")
    print(f"\n  {'Model':<28} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>7}")
    print("  " + "-" * 58)

    results = []
    trained = {}

    for name, model in models.items():
        Xtr = X_train_sc if name in SCALED_MODELS else X_train
        Xte = X_test_sc  if name in SCALED_MODELS else X_test

        model.fit(Xtr, y_train)
        yp    = model.predict(Xte)
        yprob = model.predict_proba(Xte)[:, 1]

        r = {
            "model":     name,
            "accuracy":  round(accuracy_score(y_test, yp)  * 100, 2),
            "precision": round(precision_score(y_test, yp) * 100, 2),
            "recall":    round(recall_score(y_test, yp)    * 100, 2),
            "f1":        round(f1_score(y_test, yp)        * 100, 2),
            "auc":       round(roc_auc_score(y_test, yprob), 4),
        }
        results.append(r)
        trained[name] = model

        print(f"  {name:<28} {r['accuracy']:>5.1f}% {r['precision']:>5.1f}% "
              f"{r['recall']:>5.1f}% {r['f1']:>5.1f}% {r['auc']:>7.4f}")

    # Stacking Ensemble
    print("\n  Training Stacking Ensemble (5-fold CV)...")
    stacker = StackingClassifier(
        estimators=[
            ("rf",  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_SEED)),
            ("gb",  GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=RANDOM_SEED)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                  alpha=0.01, random_state=RANDOM_SEED)),
        ],
        final_estimator=LogisticRegression(C=1.0, max_iter=500),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stacker.fit(X_train, y_train)
    yp_s    = stacker.predict(X_test)
    yprob_s = stacker.predict_proba(X_test)[:, 1]

    r_s = {
        "model":     "Stacking Ensemble",
        "accuracy":  round(accuracy_score(y_test, yp_s)  * 100, 2),
        "precision": round(precision_score(y_test, yp_s) * 100, 2),
        "recall":    round(recall_score(y_test, yp_s)    * 100, 2),
        "f1":        round(f1_score(y_test, yp_s)        * 100, 2),
        "auc":       round(roc_auc_score(y_test, yprob_s), 4),
    }
    results.append(r_s)
    trained["Stacking Ensemble"] = stacker

    print(f"  {'Stacking Ensemble':<28} {r_s['accuracy']:>5.1f}% {r_s['precision']:>5.1f}% "
          f"{r_s['recall']:>5.1f}% {r_s['f1']:>5.1f}% {r_s['auc']:>7.4f}")

    # Best model
    best = max(results, key=lambda x: x["auc"])
    print(f"\n  🏆 Best model: {best['model']} "
          f"(AUC={best['auc']}, Acc={best['accuracy']}%)")

    return results, trained


# ─── 6. FEATURE IMPORTANCE ───────────────────────────────────────────────────
def compute_feature_importance(rf_model, feature_cols: list) -> list:
    """Compute and return SHAP-proxy feature importances from Random Forest."""
    fi = list(zip(feature_cols, rf_model.feature_importances_))
    fi.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*60}")
    print("STEP 5 — FEATURE IMPORTANCE (Random Forest — SHAP proxy)")
    print(f"{'='*60}")
    for fname, fval in fi:
        bar = "█" * int(fval * 100)
        print(f"  {fname:<25} {fval:.4f}  {bar}")

    return fi


# ─── 7. SAVE ARTIFACTS ───────────────────────────────────────────────────────
def save_artifacts(best_model, scaler, results: list, fi: list,
                   feature_cols: list, best_name: str,
                   y_test, y_pred, cm):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(scaler,     f"{MODEL_DIR}/scaler.pkl")

    meta = {
        "best_model_name":   best_name,
        "feature_columns":   feature_cols,
        "best_needs_scaling": best_name in SCALED_MODELS,
        "scaled_models":     list(SCALED_MODELS),
        "results":           results,
        "feature_importance": [(f, round(float(v), 6)) for f, v in fi],
        "confusion_matrix":  cm.tolist(),
        "binary_map":        BINARY_MAP,
        "target_map":        TARGET_MAP,
        "class_report":      classification_report(y_test, y_pred, output_dict=True),
    }
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✓ Saved: {MODEL_DIR}/best_model.pkl")
    print(f"  ✓ Saved: {MODEL_DIR}/scaler.pkl")
    print(f"  ✓ Saved: {MODEL_DIR}/model_meta.json")


# ─── 8. PLOTS ────────────────────────────────────────────────────────────────
def generate_plots(models_dict, trained, results, fi, feature_cols,
                   X_test, X_test_sc, y_test, y_pred_best):

    os.makedirs(STATIC_DIR, exist_ok=True)
    colors = ["#3b82f6","#10b981","#e53e3e","#d97706","#8b5cf6","#06b6d4","#f97316","#1a4fdb"]

    # ── Plot 1: 3-panel evaluation ────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 6), facecolor="#fafcff")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=.35)

    # 1a ROC
    ax1 = fig.add_subplot(gs[0])
    for i, (name, model) in enumerate(trained.items()):
        Xte = X_test_sc if name in SCALED_MODELS else X_test
        try:
            yp  = model.predict_proba(Xte)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, yp)
            auc = roc_auc_score(y_test, yp)
            ax1.plot(fpr, tpr, color=colors[i % len(colors)], lw=1.8,
                     label=f"{name.replace(' (MLP)','').replace(' Ensemble','')[:18]} ({auc:.3f})")
        except Exception:
            pass
    ax1.plot([0,1],[0,1],"k--", alpha=.3)
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate",  fontsize=10)
    ax1.set_title("ROC Curves — All Models", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=6.5, framealpha=.8)
    ax1.grid(alpha=.25)
    ax1.set_facecolor("#f8faff")

    # 1b Accuracy comparison
    ax2 = fig.add_subplot(gs[1])
    names = [r["model"].replace("Neural Network","MLP").replace(" Ensemble","") for r in results]
    accs  = [r["accuracy"] for r in results]
    bar_colors = [colors[i % len(colors)] for i in range(len(results))]
    bar_colors[-1] = "#e53e3e"  # Highlight stacking
    bar_colors[2]  = "#1a4fdb"  # Highlight RF (best)
    bars = ax2.bar(range(len(results)), accs, color=bar_colors, width=.65, edgecolor="white", linewidth=.5)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Model Accuracy Comparison", fontsize=11, fontweight="bold")
    ax2.set_ylim(60, 103)
    ax2.grid(axis="y", alpha=.25)
    ax2.set_facecolor("#f8faff")
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .3,
                 f"{acc:.1f}%", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    # 1c Feature importance
    ax3 = fig.add_subplot(gs[2])
    fnames = [f for f, _ in fi[:10]]
    fvals  = [v for _, v in fi[:10]]
    pal    = plt.cm.Blues(np.linspace(0.35, 0.9, len(fnames)))[::-1]
    ax3.barh(range(len(fnames)), fvals[::-1], color=pal[::-1], edgecolor="white")
    ax3.set_yticks(range(len(fnames)))
    ax3.set_yticklabels(fnames[::-1], fontsize=8)
    ax3.set_xlabel("Feature Importance", fontsize=10)
    ax3.set_title("Top 10 Features (RF Importance)", fontsize=11, fontweight="bold")
    ax3.grid(axis="x", alpha=.25)
    ax3.set_facecolor("#f8faff")

    fig.suptitle("DiabetesSense v2.0 — Model Evaluation Dashboard",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(f"{STATIC_DIR}/model_evaluation.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: {STATIC_DIR}/model_evaluation.png")

    # ── Plot 2: Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred_best)
    fig2, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("Confusion Matrix — Random Forest (Best Model)",
                 fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{STATIC_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {STATIC_DIR}/confusion_matrix.png")


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  DiabetesSense v2.0 — Full ML Training Pipeline")
    print("  Dataset: Early Stage Diabetes Risk Prediction (UCI)")
    print("="*60)

    # 1. Load
    df = load_and_clean(DATA_PATH)

    # 2. Encode
    X, y, feature_cols = encode_features(df)

    # 3. Split + Scale
    (X_train, X_test, y_train, y_test,
     X_train_sc, X_test_sc, scaler) = prepare_data(X, y)

    # 4. Build & train models
    models = build_models()
    results, trained = evaluate_all(
        models, X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
    )

    # 5. Feature importance (RF proxy for SHAP)
    fi = compute_feature_importance(trained["Random Forest"], feature_cols)

    # 6. Best model details
    best_result = max(results, key=lambda x: x["auc"])
    best_name   = best_result["model"]
    best_model  = trained[best_name]

    # Predictions for CM + report
    Xte         = X_test_sc if best_name in SCALED_MODELS else X_test
    y_pred_best = best_model.predict(Xte)
    cm          = confusion_matrix(y_test, y_pred_best)

    print(f"\n{'='*60}")
    print("STEP 6 — CLASSIFICATION REPORT (Best: {})".format(best_name))
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred_best,
                                 target_names=["Negative", "Positive"]))

    # 7. Save
    save_artifacts(best_model, scaler, results, fi, feature_cols,
                   best_name, y_test, y_pred_best, cm)

    # 8. Plots
    print(f"\n{'='*60}")
    print("STEP 7 — GENERATING EVALUATION PLOTS")
    print(f"{'='*60}")
    generate_plots(models, trained, results, fi, feature_cols,
                   X_test, X_test_sc, y_test, y_pred_best)

    print(f"\n{'='*60}")
    print("✅  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best Model: {best_name}")
    print(f"  Accuracy:   {best_result['accuracy']}%")
    print(f"  AUC-ROC:    {best_result['auc']}")
    print(f"  Artifacts:  {MODEL_DIR}/")
    print(f"  Plots:      {STATIC_DIR}/")
    print(f"\n  Next step:")
    print(f"  ► uvicorn backend.main:app --reload --port 8000\n")


if __name__ == "__main__":
    main()
