"""SafeShift evaluation harness.

Compares, against an INDEPENDENT non-linear ground truth (evaluation/synthetic.py):
  * Random baseline
  * SafeShift transparent heuristic (linear expert weighting)
  * Logistic Regression (interpretable learned)
  * Random Forest (non-linear learned; SafeShift's optional model)

Reports held-out ROC-AUC, PR-AUC, Brier, and threshold metrics; 5-fold CV; an ablation by
feature group; and a label-noise robustness sweep. Saves figures and results.md / results.json.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_fscore_support, confusion_matrix, roc_curve)

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import synthetic as S
from safeshift.model import heuristic_score
from safeshift.features import INTERFACE_FEATURE_NAMES as FN

SEED = 11
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def heuristic_proba(X):
    return np.array([heuristic_score({FN[i]: row[i] for i in range(len(FN))}) for row in X])


def threshold_metrics(y, p, t=0.5):
    yhat = (p >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return dict(precision=pr, recall=rc, f1=f1, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def evaluate_all():
    rng = np.random.default_rng(SEED)
    X, y = S.make_dataset(n=8000, seed=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)

    results = {"n_total": int(len(y)), "n_test": int(len(yte)), "positive_rate": float(y.mean())}

    # models -> test probabilities
    proba = {}
    proba["Random"] = rng.random(len(yte))
    proba["Heuristic (linear)"] = heuristic_proba(Xte)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    proba["Logistic Regression"] = lr.predict_proba(Xte)[:, 1]
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED).fit(Xtr, ytr)
    proba["Random Forest"] = rf.predict_proba(Xte)[:, 1]

    metrics = {}
    for name, p in proba.items():
        m = {"roc_auc": float(roc_auc_score(yte, p)),
             "pr_auc": float(average_precision_score(yte, p)),
             "brier": float(brier_score_loss(yte, np.clip(p, 0, 1)))}
        m.update(threshold_metrics(yte, p))
        metrics[name] = m
    results["holdout_metrics"] = metrics

    # 5-fold CV (RF) for stability
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr, va in skf.split(X, y):
        m = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    results["rf_cv_auc_mean"] = float(np.mean(aucs))
    results["rf_cv_auc_std"] = float(np.std(aucs))
    results["rf_cv_auc_folds"] = [round(float(a), 4) for a in aucs]

    # ablation by feature group (zero out group, retrain RF)
    full_auc = metrics["Random Forest"]["roc_auc"]
    ablation = {}
    name_to_idx = {n: i for i, n in enumerate(FN)}
    for group, names in S.FEATURE_GROUPS.items():
        idx = [name_to_idx[n] for n in names]
        Xtr2, Xte2 = Xtr.copy(), Xte.copy()
        Xtr2[:, idx] = 0.0; Xte2[:, idx] = 0.0
        m = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED).fit(Xtr2, ytr)
        auc = roc_auc_score(yte, m.predict_proba(Xte2)[:, 1])
        ablation[group] = {"auc_without": float(auc), "auc_drop": float(full_auc - auc)}
    results["ablation"] = ablation

    # robustness: label-noise sweep (RF AUC)
    noise_curve = {}
    for noise in [0.5, 1.0, 1.5, 2.0, 3.0]:
        Xn, yn = S.make_dataset(n=6000, seed=SEED + 1, noise=noise)
        xtr, xte, ya, yb = train_test_split(Xn, yn, test_size=0.3, random_state=SEED, stratify=yn)
        m = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=SEED).fit(xtr, ya)
        noise_curve[str(noise)] = float(roc_auc_score(yb, m.predict_proba(xte)[:, 1]))
    results["noise_robustness_rf_auc"] = noise_curve

    # ---- figures ----
    plt.figure(figsize=(5, 4))
    for name in ["Random Forest", "Logistic Regression", "Heuristic (linear)", "Random"]:
        fpr, tpr, _ = roc_curve(yte, proba[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics[name]['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — interface integration-risk prediction"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "roc.png"), dpi=150); plt.close()

    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6, 4))
    plt.barh([FN[i] for i in order][::-1], importances[order][::-1], color="#2E5FA3")
    plt.xlabel("Random-forest feature importance")
    plt.title("Feature importance (learned model)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "feature_importance.png"), dpi=150); plt.close()
    results["rf_feature_importance"] = {FN[i]: float(importances[i]) for i in order}

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def write_markdown(r):
    L = []
    L.append("# SafeShift Evaluation Results\n")
    L.append(f"- Dataset: {r['n_total']} synthetic interfaces (independent non-linear ground truth); "
             f"test set {r['n_test']}; positive rate {r['positive_rate']:.2f}.\n")
    L.append("## Held-out performance (30% test set)\n")
    L.append("| Model | ROC-AUC | PR-AUC | Brier | Precision | Recall | F1 |")
    L.append("|-------|--------:|-------:|------:|----------:|-------:|---:|")
    for name, m in r["holdout_metrics"].items():
        L.append(f"| {name} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['brier']:.3f} "
                 f"| {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |")
    L.append(f"\n5-fold CV (Random Forest) ROC-AUC: **{r['rf_cv_auc_mean']:.3f} ± {r['rf_cv_auc_std']:.3f}** "
             f"(folds: {r['rf_cv_auc_folds']}).\n")
    L.append("## Ablation — ROC-AUC drop when a feature group is removed (Random Forest)\n")
    L.append("| Feature group removed | ROC-AUC without | AUC drop |")
    L.append("|-----------------------|----------------:|---------:|")
    for g, d in sorted(r["ablation"].items(), key=lambda kv: kv[1]["auc_drop"], reverse=True):
        L.append(f"| {g} | {d['auc_without']:.3f} | {d['auc_drop']:.3f} |")
    L.append("\n## Robustness — RF ROC-AUC vs label-noise temperature\n")
    L.append("| Noise (1.0=default, higher=noisier) | RF ROC-AUC |")
    L.append("|---:|---:|")
    for k, v in r["noise_robustness_rf_auc"].items():
        L.append(f"| {k} | {v:.3f} |")
    L.append("\n## Random-forest feature importance (descending)\n")
    for k, v in r["rf_feature_importance"].items():
        L.append(f"- {k}: {v:.3f}")
    L.append("\nFigures: `figures/roc.png`, `figures/feature_importance.png`.")
    open(os.path.join(HERE, "results.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    res = evaluate_all()
    write_markdown(res)
    print("ROC-AUC:", {k: round(v["roc_auc"], 3) for k, v in res["holdout_metrics"].items()})
    print("RF CV AUC: %.3f +/- %.3f" % (res["rf_cv_auc_mean"], res["rf_cv_auc_std"]))
    print("Wrote evaluation/results.md, results.json, figures/*.png")
