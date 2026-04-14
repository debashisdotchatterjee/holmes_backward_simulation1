
"""
holmes_backward_simulation.py

A Colab-friendly simulation and verification pipeline showing how to create a
Sherlock-Holmes-style detective story backward: first simulate the latent case,
then generate clues, then solve the case with a Bayesian inference engine, and
finally render plots, tables, and a narrative story summary.

Author: OpenAI assistant for Debashis Chatterjee
"""

from __future__ import annotations

import os
import math
import json
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from sklearn.metrics import confusion_matrix, log_loss, brier_score_loss


# ---------------------------- Configuration ---------------------------- #

SUSPECTS = [
    "outsider_gambler",
    "trainer",
    "rival_trainer",
    "unknown_intruder",
]

MOTIVES = {
    "outsider_gambler": ["betting_sabotage", "theft_attempt"],
    "trainer": ["concealed_financial_pressure", "betting_sabotage"],
    "rival_trainer": ["competitive_sabotage", "horse_concealment"],
    "unknown_intruder": ["opportunistic_intrusion"],
}

CONCEALERS = ["none", "rival_trainer", "outsider_gambler"]

BINARY_CLUES = [
    "stable_boy_drugged",
    "dog_silent",
    "delicate_knife_found",
    "alias_bill_found",
    "horse_at_scene",
    "coat_deliberately_placed",
    "horse_hidden_by_neighbor",
    "death_consistent_with_horse_kick",
    "outsider_seen_near_stable",
    "tracks_to_rival_property",
]

CONTINUOUS_CLUES = [
    "dog_familiarity_score",
    "instrument_delicacy_score",
]

CLUE_ORDER = [
    "outsider_seen_near_stable",
    "stable_boy_drugged",
    "dog_silent",
    "dog_familiarity_score",
    "delicate_knife_found",
    "instrument_delicacy_score",
    "alias_bill_found",
    "coat_deliberately_placed",
    "horse_at_scene",
    "death_consistent_with_horse_kick",
    "tracks_to_rival_property",
    "horse_hidden_by_neighbor",
]


@dataclass
class ModelSpec:
    priors: Dict[str, float]
    bernoulli_probs: Dict[str, Dict[str, float]]
    beta_params: Dict[str, Dict[str, Tuple[float, float]]]


def build_model_spec() -> ModelSpec:
    priors = {
        "outsider_gambler": 0.22,
        "trainer": 0.48,
        "rival_trainer": 0.18,
        "unknown_intruder": 0.12,
    }

    bernoulli_probs = {
        "outsider_gambler": {
            "stable_boy_drugged": 0.30,
            "dog_silent": 0.10,
            "delicate_knife_found": 0.06,
            "alias_bill_found": 0.03,
            "horse_at_scene": 0.60,
            "coat_deliberately_placed": 0.22,
            "horse_hidden_by_neighbor": 0.08,
            "death_consistent_with_horse_kick": 0.18,
            "outsider_seen_near_stable": 0.82,
            "tracks_to_rival_property": 0.10,
        },
        "trainer": {
            "stable_boy_drugged": 0.91,
            "dog_silent": 0.96,
            "delicate_knife_found": 0.89,
            "alias_bill_found": 0.84,
            "horse_at_scene": 0.93,
            "coat_deliberately_placed": 0.72,
            "horse_hidden_by_neighbor": 0.69,
            "death_consistent_with_horse_kick": 0.86,
            "outsider_seen_near_stable": 0.26,
            "tracks_to_rival_property": 0.64,
        },
        "rival_trainer": {
            "stable_boy_drugged": 0.36,
            "dog_silent": 0.18,
            "delicate_knife_found": 0.10,
            "alias_bill_found": 0.07,
            "horse_at_scene": 0.67,
            "coat_deliberately_placed": 0.41,
            "horse_hidden_by_neighbor": 0.86,
            "death_consistent_with_horse_kick": 0.31,
            "outsider_seen_near_stable": 0.22,
            "tracks_to_rival_property": 0.88,
        },
        "unknown_intruder": {
            "stable_boy_drugged": 0.14,
            "dog_silent": 0.06,
            "delicate_knife_found": 0.03,
            "alias_bill_found": 0.01,
            "horse_at_scene": 0.49,
            "coat_deliberately_placed": 0.14,
            "horse_hidden_by_neighbor": 0.05,
            "death_consistent_with_horse_kick": 0.09,
            "outsider_seen_near_stable": 0.54,
            "tracks_to_rival_property": 0.06,
        },
    }

    beta_params = {
        "outsider_gambler": {
            "dog_familiarity_score": (2.0, 6.5),
            "instrument_delicacy_score": (2.2, 6.0),
        },
        "trainer": {
            "dog_familiarity_score": (8.0, 2.0),
            "instrument_delicacy_score": (8.5, 2.0),
        },
        "rival_trainer": {
            "dog_familiarity_score": (2.5, 6.0),
            "instrument_delicacy_score": (3.0, 5.8),
        },
        "unknown_intruder": {
            "dog_familiarity_score": (1.8, 7.2),
            "instrument_delicacy_score": (1.6, 7.0),
        },
    }
    return ModelSpec(priors=priors, bernoulli_probs=bernoulli_probs, beta_params=beta_params)


# ---------------------------- Simulation ---------------------------- #

def choose_motive(planner: str, rng: np.random.Generator) -> str:
    choices = MOTIVES[planner]
    probs = None
    if planner == "trainer":
        probs = [0.68, 0.32]
    elif planner == "outsider_gambler":
        probs = [0.75, 0.25]
    elif planner == "rival_trainer":
        probs = [0.70, 0.30]
    else:
        probs = [1.0]
    return rng.choice(choices, p=probs)


def choose_concealer(planner: str, rng: np.random.Generator) -> str:
    if planner == "trainer":
        return rng.choice(["rival_trainer", "none"], p=[0.74, 0.26])
    if planner == "rival_trainer":
        return rng.choice(["rival_trainer", "none"], p=[0.81, 0.19])
    if planner == "outsider_gambler":
        return rng.choice(["outsider_gambler", "none"], p=[0.22, 0.78])
    return "none"


def maybe_apply_story_logic(
    planner: str,
    motive: str,
    concealer: str,
    obs: Dict[str, float],
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Inject a few story-structural dependencies so the data do not look purely
    independent even though the solver uses a factorized approximation.
    """
    # If the trainer is the planner, the classical Holmes-like signature should
    # show up more coherently.
    if planner == "trainer":
        if obs["dog_silent"] == 1 and obs["stable_boy_drugged"] == 1:
            obs["horse_at_scene"] = 1 if rng.random() < 0.96 else obs["horse_at_scene"]
            obs["death_consistent_with_horse_kick"] = 1 if rng.random() < 0.93 else obs["death_consistent_with_horse_kick"]
        if motive == "concealed_financial_pressure":
            obs["alias_bill_found"] = 1 if rng.random() < 0.94 else obs["alias_bill_found"]
            obs["delicate_knife_found"] = 1 if rng.random() < 0.94 else obs["delicate_knife_found"]

    if concealer == "rival_trainer":
        obs["horse_hidden_by_neighbor"] = 1 if rng.random() < 0.96 else obs["horse_hidden_by_neighbor"]
        obs["tracks_to_rival_property"] = 1 if rng.random() < 0.95 else obs["tracks_to_rival_property"]

    # Red-herring outsider sighting is occasionally present even in trainer-led cases.
    if planner == "trainer" and rng.random() < 0.18:
        obs["outsider_seen_near_stable"] = 1

    # If dog familiarity score is high, dog_silent is more likely.
    if obs["dog_familiarity_score"] > 0.72 and rng.random() < 0.85:
        obs["dog_silent"] = 1

    # Instrument delicacy score and knife clue should positively reinforce one another.
    if obs["instrument_delicacy_score"] > 0.70 and rng.random() < 0.87:
        obs["delicate_knife_found"] = 1

    return obs


def simulate_one_case(case_id: int, spec: ModelSpec, rng: np.random.Generator) -> Dict[str, float]:
    planner = rng.choice(SUSPECTS, p=[spec.priors[s] for s in SUSPECTS])
    motive = choose_motive(planner, rng)
    concealer = choose_concealer(planner, rng)

    obs: Dict[str, float] = {}

    # Binary clues
    for clue in BINARY_CLUES:
        p = spec.bernoulli_probs[planner][clue]
        obs[clue] = int(rng.binomial(1, p))

    # Continuous clues
    for clue in CONTINUOUS_CLUES:
        a, b = spec.beta_params[planner][clue]
        obs[clue] = float(rng.beta(a, b))

    obs = maybe_apply_story_logic(planner, motive, concealer, obs, rng)

    case = {
        "case_id": case_id,
        "true_planner": planner,
        "true_motive": motive,
        "true_concealer": concealer,
    }
    case.update(obs)
    return case


def simulate_dataset(n_cases: int, spec: ModelSpec, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [simulate_one_case(i + 1, spec, rng) for i in range(n_cases)]
    df = pd.DataFrame(rows)
    return df


# ---------------------------- Inference ---------------------------- #

def safe_log(x: float, eps: float = 1e-12) -> float:
    return math.log(max(x, eps))


def bernoulli_logpmf(x: int, p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return x * safe_log(p) + (1 - x) * safe_log(1 - p)


def beta_logpdf(x: float, a: float, b: float) -> float:
    x = min(max(float(x), 1e-6), 1 - 1e-6)
    return float(beta_dist.logpdf(x, a, b))


def posterior_for_case(row: pd.Series, spec: ModelSpec) -> Dict[str, float]:
    log_scores = {}
    for suspect in SUSPECTS:
        score = safe_log(spec.priors[suspect])
        for clue in BINARY_CLUES:
            score += bernoulli_logpmf(int(row[clue]), spec.bernoulli_probs[suspect][clue])
        for clue in CONTINUOUS_CLUES:
            a, b = spec.beta_params[suspect][clue]
            score += beta_logpdf(row[clue], a, b)
        log_scores[suspect] = score

    max_log = max(log_scores.values())
    exp_scores = {k: math.exp(v - max_log) for k, v in log_scores.items()}
    normalizer = sum(exp_scores.values())
    return {k: v / normalizer for k, v in exp_scores.items()}


def posterior_trajectory(row: pd.Series, spec: ModelSpec, clue_order: List[str]) -> pd.DataFrame:
    current_log = {s: safe_log(spec.priors[s]) for s in SUSPECTS}
    history = []

    for step, clue in enumerate(clue_order, start=1):
        for suspect in SUSPECTS:
            if clue in BINARY_CLUES:
                current_log[suspect] += bernoulli_logpmf(int(row[clue]), spec.bernoulli_probs[suspect][clue])
            elif clue in CONTINUOUS_CLUES:
                a, b = spec.beta_params[suspect][clue]
                current_log[suspect] += beta_logpdf(row[clue], a, b)

        max_log = max(current_log.values())
        exp_scores = {k: math.exp(v - max_log) for k, v in current_log.items()}
        total = sum(exp_scores.values())
        post = {k: exp_scores[k] / total for k in SUSPECTS}
        rec = {"step": step, "clue": clue}
        rec.update(post)
        history.append(rec)

    return pd.DataFrame(history)


def attach_predictions(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    posteriors = df.apply(lambda row: posterior_for_case(row, spec), axis=1)
    post_df = pd.DataFrame(list(posteriors))
    post_df.columns = [f"post_{c}" for c in post_df.columns]
    out = pd.concat([df.reset_index(drop=True), post_df.reset_index(drop=True)], axis=1)

    posterior_cols = [f"post_{s}" for s in SUSPECTS]
    out["pred_planner"] = out[posterior_cols].idxmax(axis=1).str.replace("post_", "", regex=False)
    out["pred_confidence"] = out[posterior_cols].max(axis=1)
    out["is_correct"] = (out["pred_planner"] == out["true_planner"]).astype(int)
    return out


# ---------------------------- Metrics and Tables ---------------------------- #

def multiclass_brier_score(y_true: np.ndarray, probs: np.ndarray, class_names: List[str]) -> float:
    y_onehot = np.zeros_like(probs)
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for i, y in enumerate(y_true):
        y_onehot[i, class_to_idx[y]] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def compute_performance_tables(df_pred: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    posterior_cols = [f"post_{s}" for s in SUSPECTS]
    y_true = df_pred["true_planner"].to_numpy()
    probs = df_pred[posterior_cols].to_numpy()
    y_pred = df_pred["pred_planner"].to_numpy()

    accuracy = float(np.mean(y_true == y_pred))
    top2_hits = 0
    for _, row in df_pred.iterrows():
        top2 = row[posterior_cols].sort_values(ascending=False).index[:2].str.replace("post_", "", regex=False).tolist()
        top2_hits += int(row["true_planner"] in top2)
    top2_accuracy = top2_hits / len(df_pred)

    class_to_idx = {c: i for i, c in enumerate(SUSPECTS)}
    y_true_idx = np.array([class_to_idx[y] for y in y_true], dtype=int)
    ll = float(log_loss(y_true_idx, probs, labels=np.arange(len(SUSPECTS))))
    bs = multiclass_brier_score(y_true, probs, SUSPECTS)

    summary = pd.DataFrame({
        "metric": ["planner_accuracy", "top2_accuracy", "multiclass_log_loss", "multiclass_brier_score", "mean_prediction_confidence"],
        "value": [accuracy, top2_accuracy, ll, bs, float(df_pred["pred_confidence"].mean())],
    })

    cm = confusion_matrix(y_true, y_pred, labels=SUSPECTS)
    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in SUSPECTS], columns=[f"pred_{s}" for s in SUSPECTS])

    clue_rates = (
        df_pred.groupby("true_planner")[BINARY_CLUES + CONTINUOUS_CLUES]
        .mean()
        .reindex(SUSPECTS)
    )

    importance_rows = []
    for clue in BINARY_CLUES:
        vals = []
        for s in SUSPECTS:
            p = np.clip(build_model_spec().bernoulli_probs[s][clue], 1e-6, 1 - 1e-6)
            rest = np.mean([build_model_spec().bernoulli_probs[u][clue] for u in SUSPECTS if u != s])
            rest = np.clip(rest, 1e-6, 1 - 1e-6)
            vals.append(abs(math.log(p / rest)) + abs(math.log((1 - p) / (1 - rest))))
        importance_rows.append({"clue": clue, "discriminative_score": float(np.mean(vals))})

    for clue in CONTINUOUS_CLUES:
        vals = []
        spec = build_model_spec()
        xs = np.linspace(0.01, 0.99, 120)
        for s in SUSPECTS:
            a1, b1 = spec.beta_params[s][clue]
            curve1 = beta_dist.pdf(xs, a1, b1)
            others = [beta_dist.pdf(xs, *spec.beta_params[u][clue]) for u in SUSPECTS if u != s]
            curve2 = np.mean(np.vstack(others), axis=0)
            curve1 = np.clip(curve1, 1e-9, None)
            curve2 = np.clip(curve2, 1e-9, None)
            kl = np.trapezoid(curve1 * np.log(curve1 / curve2), xs)
            vals.append(float(kl))
        importance_rows.append({"clue": clue, "discriminative_score": float(np.mean(vals))})

    importance_df = pd.DataFrame(importance_rows).sort_values("discriminative_score", ascending=False)

    return {
        "performance_summary": summary,
        "confusion_matrix": cm_df,
        "clue_prevalence_by_true_planner": clue_rates.reset_index(),
        "clue_importance": importance_df.reset_index(drop=True),
    }


# ---------------------------- Story Rendering ---------------------------- #

def pretty_label(name: str) -> str:
    return name.replace("_", " ").title()


def row_to_story(row: pd.Series) -> str:
    planner = pretty_label(row["true_planner"])
    motive = pretty_label(row["true_motive"])
    concealer = pretty_label(row["true_concealer"])

    details = []
    if row["stable_boy_drugged"] == 1:
        details.append("the stable boy was rendered unusually drowsy after supper")
    if row["dog_silent"] == 1:
        details.append("the watchdog remained eerily silent during the crucial nocturnal visit")
    if row["delicate_knife_found"] == 1:
        details.append("a fine surgical-looking knife appeared among the effects of the dead man")
    if row["alias_bill_found"] == 1:
        details.append("a private bill under an assumed name hinted at concealed financial strain")
    if row["death_consistent_with_horse_kick"] == 1:
        details.append("the fatal wound aligned better with a violent kick from the horse than with ordinary combat")
    if row["horse_hidden_by_neighbor"] == 1:
        details.append("the missing horse was later traced to a neighboring property rather than to distant thieves")
    if row["outsider_seen_near_stable"] == 1:
        details.append("an outsider had earlier been observed near the stable, creating a tempting red herring")

    if not details:
        details.append("the facts were sparse and misleading, forcing the detective to reason from small asymmetries")

    details_sentence = "; ".join(details[:6]) + "."

    story = f"""
    Case {int(row['case_id'])}.
    On a rain-darkened moor, a champion racehorse vanished in the night and a body was found at dawn.
    The hidden planner of the event was {planner}; the underlying motive was {motive}; the later concealment was linked to {concealer}.
    The clue pattern was as follows: {details_sentence}
    A Holmes-like analyst would begin with the visible scandal, discount the most theatrical explanation, and then notice that silence, delicacy, and concealed debt jointly point inward rather than outward.
    """
    return textwrap.fill(" ".join(story.split()), width=100)


# ---------------------------- Plotting ---------------------------- #

def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    figs = output_dir / "figures"
    tabs = output_dir / "tables"
    stories = output_dir / "stories"
    data = output_dir / "data"
    for d in [output_dir, figs, tabs, stories, data]:
        d.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "figs": figs, "tabs": tabs, "stories": stories, "data": data}


def save_table(df: pd.DataFrame, path_base: Path) -> None:
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    try:
        df.to_markdown(path_base.with_suffix(".md"), index=False)
    except Exception:
        pass


def plot_clue_heatmap(spec: ModelSpec, fig_path: Path, show: bool) -> None:
    clue_names = BINARY_CLUES + CONTINUOUS_CLUES
    values = np.zeros((len(SUSPECTS), len(clue_names)))
    for i, s in enumerate(SUSPECTS):
        for j, c in enumerate(BINARY_CLUES):
            values[i, j] = spec.bernoulli_probs[s][c]
        for j, c in enumerate(CONTINUOUS_CLUES, start=len(BINARY_CLUES)):
            a, b = spec.beta_params[s][c]
            values[i, j] = a / (a + b)

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(len(clue_names)))
    ax.set_yticks(np.arange(len(SUSPECTS)))
    ax.set_xticklabels([pretty_label(c) for c in clue_names], rotation=70, ha="right")
    ax.set_yticklabels([pretty_label(s) for s in SUSPECTS])
    ax.set_title("Model-Encoded Mean Clue Intensity by Hidden Planner")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability or Mean Score")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_heatmap(cm_df: pd.DataFrame, fig_path: Path, show: bool) -> None:
    values = cm_df.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_df.index)
    ax.set_title("Confusion Matrix for Hidden Planner Recovery")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_posterior_trajectory(traj: pd.DataFrame, fig_path: Path, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for suspect in SUSPECTS:
        ax.plot(traj["step"], traj[suspect], marker="o", linewidth=2, label=pretty_label(suspect))
    ax.set_xticks(traj["step"])
    ax.set_xticklabels([f"{i}\n{c}" for i, c in zip(traj["step"], traj["clue"])], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Posterior Probability")
    ax.set_title("Sequential Holmesian Posterior Updating for the Showcase Case")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_reliability(df_pred: pd.DataFrame, fig_path: Path, show: bool) -> None:
    conf = df_pred["pred_confidence"].to_numpy()
    correct = df_pred["is_correct"].to_numpy()

    bins = np.linspace(0, 1, 11)
    ids = np.digitize(conf, bins, right=True)
    centers, accs, counts = [], [], []
    for b in range(1, len(bins)):
        mask = ids == b
        if mask.sum() == 0:
            continue
        centers.append(conf[mask].mean())
        accs.append(correct[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6.8, 5.3))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Ideal calibration")
    ax.plot(centers, accs, marker="o", linewidth=2, label="Observed accuracy")
    for x, y, n in zip(centers, accs, counts):
        ax.text(x, y, f"n={n}", fontsize=8, va="bottom")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability of the Holmesian Posterior")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_importance(importance_df: pd.DataFrame, fig_path: Path, show: bool) -> None:
    temp = importance_df.sort_values("discriminative_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.barh(temp["clue"], temp["discriminative_score"])
    ax.set_xlabel("Discriminative score")
    ax.set_ylabel("Clue")
    ax.set_title("Which Clues Most Sharply Separate Rival Explanations?")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_true_class_posterior_distribution(df_pred: pd.DataFrame, fig_path: Path, show: bool) -> None:
    vals = []
    for _, row in df_pred.iterrows():
        vals.append(row[f"post_{row['true_planner']}"])
    vals = np.asarray(vals)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.hist(vals, bins=20, edgecolor="black")
    ax.set_xlabel("Posterior assigned to the true hidden planner")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Posterior Mass on the True Explanation")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------- Main Pipeline ---------------------------- #

def run_pipeline(
    n_cases: int = 1500,
    seed: int = 123,
    showcase_case_id: int | None = None,
    output_root: str = "holmes_simulation_outputs",
    show_plots: bool = True,
    show_tables: bool = True,
) -> Dict[str, str]:
    spec = build_model_spec()

    output_dir = Path(output_root)
    dirs = ensure_dirs(output_dir)

    df = simulate_dataset(n_cases=n_cases, spec=spec, seed=seed)
    df_pred = attach_predictions(df, spec)
    tables = compute_performance_tables(df_pred)

    # Save raw and enriched data
    df.to_csv(dirs["data"] / "simulated_cases.csv", index=False)
    df_pred.to_csv(dirs["data"] / "simulated_cases_with_posteriors.csv", index=False)

    # Save parameter tables
    bern_rows = []
    for suspect in SUSPECTS:
        rec = {"suspect": suspect}
        rec.update(spec.bernoulli_probs[suspect])
        bern_rows.append(rec)
    bern_df = pd.DataFrame(bern_rows)
    beta_rows = []
    for suspect in SUSPECTS:
        rec = {"suspect": suspect}
        for clue in CONTINUOUS_CLUES:
            a, b = spec.beta_params[suspect][clue]
            rec[f"{clue}_alpha"] = a
            rec[f"{clue}_beta"] = b
            rec[f"{clue}_mean"] = a / (a + b)
        beta_rows.append(rec)
    beta_df = pd.DataFrame(beta_rows)

    save_table(bern_df, dirs["tabs"] / "bernoulli_clue_parameters")
    save_table(beta_df, dirs["tabs"] / "continuous_clue_parameters")

    for name, table in tables.items():
        save_table(table, dirs["tabs"] / name)

    # Showcase case: by default, pick the most Holmes-like trainer-led case.
    if showcase_case_id is None:
        trainer_pool = df_pred[
            (df_pred["true_planner"] == "trainer")
            & (df_pred["dog_silent"] == 1)
            & (df_pred["delicate_knife_found"] == 1)
            & (df_pred["alias_bill_found"] == 1)
            & (df_pred["death_consistent_with_horse_kick"] == 1)
        ].copy()
        if len(trainer_pool) > 0:
            trainer_pool = trainer_pool.sort_values("pred_confidence", ascending=False)
            showcase_case_id = int(trainer_pool.iloc[0]["case_id"])
        else:
            showcase_case_id = int(df_pred.sort_values("pred_confidence", ascending=False).iloc[0]["case_id"])
    elif showcase_case_id not in set(df_pred["case_id"].tolist()):
        showcase_case_id = int(df_pred.iloc[0]["case_id"])
    showcase_row = df_pred.loc[df_pred["case_id"] == showcase_case_id].iloc[0]
    showcase_story = row_to_story(showcase_row)
    (dirs["stories"] / "showcase_story.txt").write_text(showcase_story, encoding="utf-8")
    traj = posterior_trajectory(showcase_row, spec, CLUE_ORDER)
    save_table(traj, dirs["tabs"] / "showcase_case_posterior_trajectory")

    # Plots
    plot_clue_heatmap(spec, dirs["figs"] / "figure_01_clue_heatmap.png", show_plots)
    plot_confusion_heatmap(tables["confusion_matrix"].set_index(tables["confusion_matrix"].columns[0]) if False else tables["confusion_matrix"], dirs["figs"] / "figure_02_confusion_matrix.png", show_plots)
    plot_posterior_trajectory(traj, dirs["figs"] / "figure_03_showcase_posterior_trajectory.png", show_plots)
    plot_reliability(df_pred, dirs["figs"] / "figure_04_reliability.png", show_plots)
    plot_importance(tables["clue_importance"], dirs["figs"] / "figure_05_clue_importance.png", show_plots)
    plot_true_class_posterior_distribution(df_pred, dirs["figs"] / "figure_06_true_class_posterior.png", show_plots)

    # Print inline outputs
    if show_tables:
        print("\n=== PERFORMANCE SUMMARY ===")
        print(tables["performance_summary"].to_string(index=False))
        print("\n=== CONFUSION MATRIX ===")
        print(tables["confusion_matrix"].to_string())
        print("\n=== CLUE IMPORTANCE ===")
        print(tables["clue_importance"].to_string(index=False))
        print("\n=== SHOWCASE STORY ===")
        print(showcase_story)
        print("\n=== FIRST 8 SIMULATED CASES ===")
        display_cols = ["case_id", "true_planner", "true_motive", "true_concealer"] + BINARY_CLUES[:5] + CONTINUOUS_CLUES
        print(df_pred[display_cols].head(8).to_string(index=False))

    # Save concise run metadata
    meta = {
        "n_cases": int(n_cases),
        "seed": int(seed),
        "showcase_case_id": int(showcase_case_id),
        "accuracy": float(tables["performance_summary"].loc[tables["performance_summary"]["metric"] == "planner_accuracy", "value"].iloc[0]),
    }
    (dirs["root"] / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Zip output directory
    zip_base = str(dirs["root"].resolve())
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=dirs["root"])

    return {
        "output_dir": str(dirs["root"].resolve()),
        "zip_path": zip_path,
        "story_path": str((dirs["stories"] / "showcase_story.txt").resolve()),
        "data_path": str((dirs["data"] / "simulated_cases_with_posteriors.csv").resolve()),
    }


if __name__ == "__main__":
    # Colab-friendly default run.
    paths = run_pipeline(
        n_cases=1500,
        seed=123,
        showcase_case_id=None,
        output_root="holmes_simulation_outputs",
        show_plots=True,
        show_tables=True,
    )
    print("\nSaved artifacts:")
    for k, v in paths.items():
        print(f"{k}: {v}")
