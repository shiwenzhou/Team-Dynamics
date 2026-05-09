"""Synthetic bio-behavioral team dynamics analytics for a Streamlit chatbot prototype.

This module supports a course prototype connected to the dissertation RQ2 goal:
building a customized AI bot that helps non-experts understand and interpret
bio-behavioral team dynamics analytics. It uses synthetic data only. It does not
use BioTDMS data, does not test dissertation hypotheses, and does not evaluate
real trainees.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROLES: List[str] = ["FiST Leader", "FSO", "JTAC", "FOA", "FOM"]
ROLE_SLUGS: List[str] = ["fist_leader", "fso", "jtac", "foa", "fom"]
ROLE_SLUG_TO_NAME: Dict[str, str] = dict(zip(ROLE_SLUGS, ROLES))
ROLE_STATE_COLUMNS: List[str] = [f"{slug}_state" for slug in ROLE_SLUGS]

STATE_LABELS: Dict[int, str] = {
    0: "monitoring",
    1: "planning",
    2: "coordinating",
    3: "executing",
    4: "reorganizing",
}
STATE_CODES: List[int] = list(STATE_LABELS.keys())
DEFAULT_USER_QUESTION = (
    "What do the synthetic entropy trajectory, inverse sample entropy trajectory, "
    "and AMI influence profile show?"
)


@dataclass(frozen=True)
class OutputPaths:
    """Paths produced by the prototype pipeline."""

    root: Path
    figures: Path
    tables: Path
    synthetic_csv: Path
    state_csv: Path
    entropy_csv: Path
    inverse_sampen_csv: Path
    role_ami_long_csv: Path
    role_ami_summary_csv: Path
    team_state_figure: Path
    entropy_figure: Path
    inverse_sampen_figure: Path
    role_ami_figure: Path
    role_influence_heatmap: Path
    packet_json: Path
    prompt_md: Path
    api_status_json: Path
    api_response_md: Path


def build_output_paths(output_dir: str | Path = "outputs") -> OutputPaths:
    """Create output folders and return an organized set of output paths."""

    root = Path(output_dir).resolve()
    figures = root / "figures"
    tables = root / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        root=root,
        figures=figures,
        tables=tables,
        synthetic_csv=root / "synthetic_biobehavioral_timeseries.csv",
        state_csv=root / "synthetic_symbolic_states.csv",
        entropy_csv=tables / "moving_window_entropy.csv",
        inverse_sampen_csv=tables / "moving_window_inverse_sample_entropy.csv",
        role_ami_long_csv=tables / "moving_window_role_ami_long.csv",
        role_ami_summary_csv=tables / "role_ami_summary.csv",
        team_state_figure=figures / "synthetic_team_state_trajectory.png",
        entropy_figure=figures / "synthetic_entropy_trajectory.png",
        inverse_sampen_figure=figures / "synthetic_inverse_sampen_trajectory.png",
        role_ami_figure=figures / "synthetic_role_ami_summary.png",
        role_influence_heatmap=figures / "synthetic_role_influence_heatmap.png",
        packet_json=root / "team_dynamics_explanation_packet.json",
        prompt_md=root / "gemini_prompt_payload.md",
        api_status_json=root / "api_status.json",
        api_response_md=root / "gemini_chatbot_response.md",
    )


def _phase_label(t: int, n_seconds: int) -> str:
    """Return a scenario-like phase label for a synthetic timestamp."""

    if t < 0.22 * n_seconds:
        return "orientation"
    if t < 0.47 * n_seconds:
        return "fire planning"
    if t < 0.78 * n_seconds:
        return "execution"
    return "assessment"


def _latent_state_for_phase(phase: str) -> int:
    """Map a scenario phase to a baseline symbolic state."""

    return {
        "orientation": 0,
        "fire planning": 1,
        "execution": 3,
        "assessment": 2,
    }.get(phase, 0)


def _bounded_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> float:
    """Draw a clipped normal value."""

    return float(np.clip(rng.normal(mean, sd), low, high))


def generate_synthetic_biobehavioral_timeseries(
    n_seconds: int = 900,
    seed: int = 42,
    event_time_sec: int = 450,
) -> pd.DataFrame:
    """Generate a synthetic 1 Hz multimodal team time series.

    The generated data are not intended to model real physiology. They are a
    safe, public demonstration dataset with enough structure to test the
    analysis and explanation workflow. Each row is one second. For each of five
    Fire Support Team roles, the function creates simple continuous features
    that resemble analysis-ready summaries: heart rate, respiration rate, EEG
    alpha power, gaze-task focus, and communication activity. The function also
    creates symbolic role states and a symbolic team state used by the metric
    functions.
    """

    if n_seconds < 300:
        raise ValueError("n_seconds must be at least 300 for stable moving-window metrics.")
    if not 30 <= event_time_sec <= n_seconds - 30:
        raise ValueError("event_time_sec must be at least 30 seconds from each edge.")

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    previous_team_state = 0
    run_id = f"synthetic_seed{seed}_n{n_seconds}_event{event_time_sec}"

    role_parameters = {
        "fist_leader": {"hr": 82, "resp": 15.5, "alpha": 0.46, "focus": 0.70, "comm": 0.18, "bias": 2},
        "fso": {"hr": 78, "resp": 14.8, "alpha": 0.50, "focus": 0.64, "comm": 0.14, "bias": 1},
        "jtac": {"hr": 86, "resp": 16.2, "alpha": 0.42, "focus": 0.73, "comm": 0.20, "bias": 3},
        "foa": {"hr": 76, "resp": 14.3, "alpha": 0.53, "focus": 0.60, "comm": 0.11, "bias": 0},
        "fom": {"hr": 75, "resp": 14.1, "alpha": 0.55, "focus": 0.58, "comm": 0.10, "bias": 0},
    }

    for t in range(n_seconds):
        phase = _phase_label(t, n_seconds)
        base_state = _latent_state_for_phase(phase)
        distance = abs(t - event_time_sec)
        event_zone = distance <= 75
        post_event = t >= event_time_sec

        # Scenario demand is a smooth latent variable that increases around the
        # synthetic event and remains moderately elevated afterward.
        event_bump = math.exp(-((t - event_time_sec) ** 2) / (2 * 55**2))
        phase_demand = {"orientation": 0.15, "fire planning": 0.35, "execution": 0.70, "assessment": 0.40}[phase]
        scenario_demand = float(np.clip(phase_demand + 0.45 * event_bump, 0, 1))

        # State variability increases around the event to create a visible
        # reorganization/adaptation signal for the entropy trajectory.
        if event_zone:
            state_variability = 0.50
        elif post_event:
            state_variability = 0.28
        else:
            state_variability = 0.14

        role_states: Dict[str, int] = {}
        continuous: Dict[str, float] = {}
        for slug, params in role_parameters.items():
            # Structured role-state generation. JTAC and FiST Leader become more
            # strongly tied to the changing team state around the synthetic event.
            if event_zone and slug in {"jtac", "fist_leader"}:
                state = int(rng.choice([2, 3, 4], p=[0.34, 0.33, 0.33]))
            elif rng.random() < state_variability:
                state = int(rng.choice(STATE_CODES))
            elif rng.random() < 0.22:
                state = int(params["bias"])
            else:
                state = int(base_state)
            role_states[f"{slug}_state"] = state

            state_load = state / max(STATE_CODES)
            coordination_boost = 0.10 if state in {2, 4} else 0.0
            execution_boost = 0.08 if state == 3 else 0.0
            communication_spike = 0.18 if event_zone and slug in {"jtac", "fist_leader"} else 0.0

            continuous[f"{slug}_heart_rate"] = _bounded_normal(
                rng,
                params["hr"] + 10.0 * scenario_demand + 3.0 * execution_boost,
                2.2,
                55,
                140,
            )
            continuous[f"{slug}_respiration_rate"] = _bounded_normal(
                rng,
                params["resp"] + 3.8 * scenario_demand + 1.2 * execution_boost,
                0.8,
                8,
                32,
            )
            continuous[f"{slug}_eeg_alpha_power"] = _bounded_normal(
                rng,
                params["alpha"] - 0.10 * scenario_demand + 0.03 * (1 - state_load),
                0.035,
                0.05,
                0.95,
            )
            continuous[f"{slug}_gaze_task_focus"] = _bounded_normal(
                rng,
                params["focus"] + 0.08 * scenario_demand - 0.04 * coordination_boost,
                0.06,
                0.05,
                0.98,
            )
            continuous[f"{slug}_communication_activity"] = _bounded_normal(
                rng,
                params["comm"] + 0.16 * coordination_boost + 0.08 * scenario_demand + communication_spike,
                0.05,
                0.0,
                1.0,
            )

        # Synthetic team state: weighted role pattern plus inertia. This is a
        # generated label used for public demonstration; it is not a BioTDMS output.
        weights = {
            "fist_leader_state": 1.25,
            "fso_state": 0.95,
            "jtac_state": 1.20,
            "foa_state": 0.85,
            "fom_state": 0.85,
        }
        weighted_counts: Dict[int, float] = {code: 0.0 for code in STATE_CODES}
        for col, state in role_states.items():
            weighted_counts[state] += weights[col]
        if rng.random() < 0.10 and not event_zone:
            team_state = previous_team_state
        else:
            team_state = max(weighted_counts.items(), key=lambda item: item[1])[0]
        previous_team_state = int(team_state)

        row: Dict[str, Any] = {
            "time_sec": int(t),
            "time_min": round(t / 60, 4),
            "run_id": run_id,
            "phase": phase,
            "event_marker": "synthetic_task_change" if t == event_time_sec else "",
            "event_zone": bool(event_zone),
            "scenario_demand": round(scenario_demand, 4),
            "team_state": int(team_state),
            "team_state_label": STATE_LABELS[int(team_state)],
        }
        row.update(role_states)
        row.update({key: round(value, 4) for key, value in continuous.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def extract_symbolic_state_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return only timestamp, phase, event, team state, and role state columns."""

    validate_timeseries_dataframe(df)
    keep = ["time_sec", "time_min", "run_id", "phase", "event_marker", "event_zone", "team_state", "team_state_label"]
    keep.extend(ROLE_STATE_COLUMNS)
    return df[keep].copy()


def validate_timeseries_dataframe(df: pd.DataFrame) -> None:
    """Validate the schema required by the metric functions."""

    required = {"time_sec", "time_min", "phase", "team_state", *ROLE_STATE_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("The input dataframe is empty.")
    if not df["time_sec"].is_monotonic_increasing:
        raise ValueError("time_sec must be monotonically increasing.")
    invalid_states = set(df["team_state"].unique()).difference(STATE_CODES)
    if invalid_states:
        raise ValueError(f"Unexpected team_state values: {sorted(invalid_states)}")


def shannon_entropy(values: Sequence[Any], base: float = 2.0) -> float:
    """Compute Shannon entropy for a discrete sequence."""

    seq = list(values)
    if not seq:
        return float("nan")
    counts = Counter(seq)
    n = len(seq)
    return float(-sum((count / n) * math.log(count / n, base) for count in counts.values()))


def normalized_shannon_entropy(values: Sequence[Any], possible_states: int) -> float:
    """Compute normalized Shannon entropy in [0, 1] when possible."""

    if possible_states <= 1:
        return 0.0
    h = shannon_entropy(values, base=2.0)
    max_h = math.log(possible_states, 2.0)
    return float(h / max_h) if max_h else 0.0


def moving_window_entropy(
    df: pd.DataFrame,
    state_col: str = "team_state",
    window: int = 120,
    step: int = 15,
) -> pd.DataFrame:
    """Compute moving-window Shannon entropy for a symbolic state sequence."""

    validate_timeseries_dataframe(df)
    if window <= 5 or step <= 0:
        raise ValueError("window must be > 5 and step must be > 0.")
    if window > len(df):
        raise ValueError("window cannot be larger than the dataframe length.")

    rows: List[Dict[str, Any]] = []
    possible_states = len(STATE_CODES)
    for start in range(0, len(df) - window + 1, step):
        stop = start + window
        segment = df.iloc[start:stop]
        values = segment[state_col].tolist()
        rows.append(
            {
                "window_start_sec": int(segment["time_sec"].iloc[0]),
                "window_end_sec": int(segment["time_sec"].iloc[-1]),
                "window_mid_sec": float(segment["time_sec"].mean()),
                "window_mid_min": float(segment["time_min"].mean()),
                "entropy_bits": shannon_entropy(values, base=2.0),
                "normalized_entropy": normalized_shannon_entropy(values, possible_states),
                "unique_states": int(len(set(values))),
                "event_in_window": bool(segment["event_marker"].astype(bool).any()),
            }
        )
    return pd.DataFrame(rows)


def _categorical_templates(seq: Sequence[Any], m: int) -> List[Tuple[Any, ...]]:
    """Create overlapping categorical templates of length m."""

    return [tuple(seq[i : i + m]) for i in range(0, len(seq) - m + 1)]


def sample_entropy_categorical(seq: Sequence[Any], m: int = 2) -> float:
    """Compute categorical sample entropy using exact template matches.

    SampEn is -ln(A/B), where B is the number of non-self matching template
    pairs of length m and A is the number of non-self matching template pairs of
    length m + 1. If no length-m matches exist, the value is undefined. If
    length-m matches exist but no length-(m+1) matches exist, SampEn is infinite
    and the inverse index is treated as 0 downstream.
    """

    values = list(seq)
    if len(values) < m + 2:
        return float("nan")

    templates_m = _categorical_templates(values, m)
    templates_mp1 = _categorical_templates(values, m + 1)

    b_count = 0
    for i in range(len(templates_m)):
        for j in range(i + 1, len(templates_m)):
            if templates_m[i] == templates_m[j]:
                b_count += 1

    a_count = 0
    for i in range(len(templates_mp1)):
        for j in range(i + 1, len(templates_mp1)):
            if templates_mp1[i] == templates_mp1[j]:
                a_count += 1

    if b_count == 0:
        return float("nan")
    if a_count == 0:
        return float("inf")
    return float(-math.log(a_count / b_count))


def inverse_sample_entropy_value(sampen: float) -> float:
    """Convert SampEn into a bounded inverse index where larger means more regular."""

    if math.isnan(sampen):
        return float("nan")
    if math.isinf(sampen):
        return 0.0
    return float(1.0 / (1.0 + max(0.0, sampen)))


def moving_window_inverse_sample_entropy(
    df: pd.DataFrame,
    state_col: str = "team_state",
    window: int = 180,
    step: int = 15,
    m: int = 2,
) -> pd.DataFrame:
    """Compute moving-window categorical SampEn and inverse SampEn."""

    validate_timeseries_dataframe(df)
    if window <= m + 2 or step <= 0:
        raise ValueError("window must be larger than m + 2 and step must be > 0.")
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(df) - window + 1, step):
        stop = start + window
        segment = df.iloc[start:stop]
        sampen = sample_entropy_categorical(segment[state_col].tolist(), m=m)
        rows.append(
            {
                "window_start_sec": int(segment["time_sec"].iloc[0]),
                "window_end_sec": int(segment["time_sec"].iloc[-1]),
                "window_mid_sec": float(segment["time_sec"].mean()),
                "window_mid_min": float(segment["time_min"].mean()),
                "sample_entropy": sampen,
                "inverse_sample_entropy": inverse_sample_entropy_value(sampen),
                "event_in_window": bool(segment["event_marker"].astype(bool).any()),
            }
        )
    return pd.DataFrame(rows)


def mutual_information_discrete(x: Sequence[Any], y: Sequence[Any], base: float = 2.0) -> float:
    """Compute mutual information I(X;Y) for two discrete sequences."""

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) == 0:
        return float("nan")
    x_values = list(x)
    y_values = list(y)
    n = len(x_values)
    joint_counts = Counter(zip(x_values, y_values))
    x_counts = Counter(x_values)
    y_counts = Counter(y_values)
    mi = 0.0
    for (x_val, y_val), joint_count in joint_counts.items():
        p_xy = joint_count / n
        p_x = x_counts[x_val] / n
        p_y = y_counts[y_val] / n
        mi += p_xy * math.log(p_xy / (p_x * p_y), base)
    return float(max(mi, 0.0))


def normalized_mutual_information(x: Sequence[Any], y: Sequence[Any]) -> float:
    """Normalize mutual information by sqrt(H(X)H(Y))."""

    h_x = shannon_entropy(x, base=2.0)
    h_y = shannon_entropy(y, base=2.0)
    denominator = math.sqrt(h_x * h_y) if h_x > 0 and h_y > 0 else 0.0
    if denominator == 0.0:
        return 0.0
    return float(mutual_information_discrete(x, y, base=2.0) / denominator)


def average_mutual_information(
    role_values: Sequence[Any],
    team_values: Sequence[Any],
    max_lag: int = 5,
) -> float:
    """Compute lag-averaged normalized mutual information.

    A positive lag compares the role sequence at time t with the team-state
    sequence at time t + lag. This supports an influence-distribution summary,
    but it remains a descriptive coupling metric and should not be interpreted
    as causal influence.
    """

    role = list(role_values)
    team = list(team_values)
    if len(role) != len(team):
        raise ValueError("role_values and team_values must have the same length.")
    scores: List[float] = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x = role
            y = team
        else:
            x = role[:-lag]
            y = team[lag:]
        if len(x) > 2:
            scores.append(normalized_mutual_information(x, y))
    return float(np.nanmean(scores)) if scores else float("nan")


def moving_window_role_ami(
    df: pd.DataFrame,
    window: int = 180,
    step: int = 15,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Compute moving-window role-level AMI and relative influence shares."""

    validate_timeseries_dataframe(df)
    if window <= max_lag + 5 or step <= 0:
        raise ValueError("window must be larger than max_lag + 5 and step must be > 0.")
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(df) - window + 1, step):
        stop = start + window
        segment = df.iloc[start:stop]
        raw_scores: Dict[str, float] = {}
        for col in ROLE_STATE_COLUMNS:
            role_name = ROLE_SLUG_TO_NAME[col.replace("_state", "")]
            raw_scores[role_name] = average_mutual_information(segment[col], segment["team_state"], max_lag=max_lag)
        total = float(np.nansum(list(raw_scores.values())))
        hhi = float(np.nansum([(score / total) ** 2 for score in raw_scores.values()])) if total > 0 else float("nan")
        for role_name, score in raw_scores.items():
            rows.append(
                {
                    "window_start_sec": int(segment["time_sec"].iloc[0]),
                    "window_end_sec": int(segment["time_sec"].iloc[-1]),
                    "window_mid_sec": float(segment["time_sec"].mean()),
                    "window_mid_min": float(segment["time_min"].mean()),
                    "role": role_name,
                    "ami": float(score),
                    "ami_share": float(score / total) if total > 0 else float("nan"),
                    "influence_concentration_hhi": hhi,
                    "event_in_window": bool(segment["event_marker"].astype(bool).any()),
                }
            )
    return pd.DataFrame(rows)


def summarize_role_ami(ami_long_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize role-level AMI across windows."""

    if ami_long_df.empty:
        raise ValueError("AMI dataframe is empty.")
    summary = (
        ami_long_df.groupby("role", as_index=False)
        .agg(mean_ami=("ami", "mean"), mean_ami_share=("ami_share", "mean"), max_ami_share=("ami_share", "max"))
        .sort_values("mean_ami_share", ascending=False)
    )
    return summary


def _event_time_from_df(df: pd.DataFrame) -> Optional[int]:
    events = df.loc[df["event_marker"].astype(bool), "time_sec"]
    if events.empty:
        return None
    return int(events.iloc[0])


def plot_team_state(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot the synthetic symbolic team state over time."""

    event_time = _event_time_from_df(df)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df["time_min"], df["team_state"], linewidth=1.2)
    if event_time is not None:
        ax.axvline(event_time / 60, linestyle="--", linewidth=1)
        ax.text(event_time / 60, max(STATE_CODES), " synthetic event", rotation=90, va="top")
    ax.set_title("Synthetic symbolic team-state trajectory")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Team state code")
    ax.set_yticks(STATE_CODES)
    ax.set_yticklabels([STATE_LABELS[i] for i in STATE_CODES])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_entropy(entropy_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot moving-window Shannon entropy."""

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(entropy_df["window_mid_min"], entropy_df["entropy_bits"], marker="o", markersize=3, linewidth=1.2)
    event_rows = entropy_df[entropy_df["event_in_window"]]
    if not event_rows.empty:
        event_mid = event_rows["window_mid_min"].median()
        ax.axvline(event_mid, linestyle="--", linewidth=1)
        ax.text(event_mid, entropy_df["entropy_bits"].max(), " event window", rotation=90, va="top")
    ax.set_title("Synthetic moving-window Shannon entropy")
    ax.set_xlabel("Window midpoint (minutes)")
    ax.set_ylabel("Entropy (bits)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_inverse_sampen(inverse_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot moving-window inverse sample entropy."""

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(inverse_df["window_mid_min"], inverse_df["inverse_sample_entropy"], marker="o", markersize=3, linewidth=1.2)
    event_rows = inverse_df[inverse_df["event_in_window"]]
    if not event_rows.empty:
        event_mid = event_rows["window_mid_min"].median()
        ax.axvline(event_mid, linestyle="--", linewidth=1)
        ax.text(event_mid, inverse_df["inverse_sample_entropy"].max(), " event window", rotation=90, va="top")
    ax.set_title("Synthetic moving-window inverse sample entropy")
    ax.set_xlabel("Window midpoint (minutes)")
    ax.set_ylabel("Inverse sample entropy index")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_role_ami_summary(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot average role-level AMI share."""

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(summary_df["role"], summary_df["mean_ami_share"])
    ax.set_title("Synthetic role-level AMI share")
    ax.set_xlabel("Role")
    ax.set_ylabel("Mean AMI share")
    ax.set_ylim(0, max(0.35, float(summary_df["mean_ami_share"].max()) * 1.25))
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_role_influence_heatmap(ami_long_df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot moving-window role influence shares as a heatmap."""

    pivot = ami_long_df.pivot_table(index="role", columns="window_mid_min", values="ami_share", aggfunc="mean")
    pivot = pivot.reindex(ROLES)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(pivot.values, aspect="auto")
    ax.set_title("Synthetic moving-window AMI share by role")
    ax.set_xlabel("Window midpoint (minutes)")
    ax.set_ylabel("Role")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    x_positions = np.linspace(0, max(0, len(pivot.columns) - 1), num=min(8, len(pivot.columns)), dtype=int)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{pivot.columns[i]:.1f}" for i in x_positions], rotation=0)
    fig.colorbar(image, ax=ax, label="AMI share")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_explanation_packet(
    timeseries_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    inverse_df: pd.DataFrame,
    ami_long_df: pd.DataFrame,
    ami_summary_df: pd.DataFrame,
    output_paths: OutputPaths,
    user_question: str = DEFAULT_USER_QUESTION,
) -> Dict[str, Any]:
    """Build a compact JSON packet for the LLM explanation prompt."""

    peak_entropy = entropy_df.loc[entropy_df["entropy_bits"].idxmax()]
    peak_inverse = inverse_df.loc[inverse_df["inverse_sample_entropy"].idxmax()]
    top_role = ami_summary_df.iloc[0]
    hhi_by_window = ami_long_df.groupby("window_mid_min", as_index=False)["influence_concentration_hhi"].mean()
    hhi_mean = float(hhi_by_window["influence_concentration_hhi"].mean())
    hhi_peak = float(hhi_by_window["influence_concentration_hhi"].max())
    event_time = _event_time_from_df(timeseries_df)

    return {
        "run_id": str(timeseries_df["run_id"].iloc[0]) if "run_id" in timeseries_df.columns and not timeseries_df.empty else "synthetic_run_001",
        "data_status": "Synthetic demonstration data only; not BioTDMS data and not dissertation results.",
        "rq2_definition_for_course_reader": (
            "RQ2 asks whether a customized AI bot can help non-experts understand and interpret "
            "bio-behavioral team dynamics analytics, including how metrics are derived and how "
            "they lead to reported results."
        ),
        "prototype_purpose": (
            "Streamlit-based chatbot scaffold for explaining interdependence, adaptation, "
            "and influence-distribution outputs from synthetic demonstration data."
        ),
        "target_constructs": ["interdependence", "adaptation", "influence distribution"],
        "user_question": user_question,
        "synthetic_time_series": {
            "sampling_rate_hz": 1,
            "n_seconds": int(len(timeseries_df)),
            "roles": ROLES,
            "modalities_per_role": [
                "heart_rate",
                "respiration_rate",
                "eeg_alpha_power",
                "gaze_task_focus",
                "communication_activity",
            ],
            "symbolic_states": STATE_LABELS,
        },
        "event_markers": [
            {"time_sec": int(event_time), "time_min": round(float(event_time) / 60, 3), "label": "Synthetic task change"}
        ]
        if event_time is not None
        else [],
        "adaptation_entropy_summary": {
            "mean_entropy_bits": round(float(entropy_df["entropy_bits"].mean()), 3),
            "peak_entropy_bits": round(float(peak_entropy["entropy_bits"]), 3),
            "peak_time_min": round(float(peak_entropy["window_mid_min"]), 3),
            "interpretation_caution": (
                "Higher entropy may indicate increased variability or reorganization, but it is not automatically better or worse."
            ),
        },
        "interdependence_inverse_sample_entropy_summary": {
            "mean_inverse_sample_entropy": round(float(inverse_df["inverse_sample_entropy"].mean()), 3),
            "peak_inverse_sample_entropy": round(float(peak_inverse["inverse_sample_entropy"]), 3),
            "peak_time_min": round(float(peak_inverse["window_mid_min"]), 3),
            "interpretation_caution": (
                "Higher inverse sample entropy indicates a more regular symbolic sequence in this synthetic example; interpretation depends on task phase and coding choices."
            ),
        },
        "influence_distribution_summary": {
            "top_role_by_mean_ami_share": str(top_role["role"]),
            "top_role_mean_ami_share": round(float(top_role["mean_ami_share"]), 3),
            "mean_hhi": round(hhi_mean, 3),
            "peak_hhi": round(hhi_peak, 3),
            "profile_note": (
                "AMI share describes descriptive coupling between each role-state sequence and the team-state sequence; it is not an individual ranking or causal claim."
            ),
        },
        "figures": {
            "team_state_trajectory": str(output_paths.team_state_figure.relative_to(output_paths.root.parent)),
            "entropy_trajectory": str(output_paths.entropy_figure.relative_to(output_paths.root.parent)),
            "inverse_sample_entropy_trajectory": str(output_paths.inverse_sampen_figure.relative_to(output_paths.root.parent)),
            "role_ami_summary": str(output_paths.role_ami_figure.relative_to(output_paths.root.parent)),
            "role_influence_heatmap": str(output_paths.role_influence_heatmap.relative_to(output_paths.root.parent)),
        },
        "requested_output": [
            "brief answer",
            "what the visualization shows",
            "metric derivation in plain language",
            "interpretation connected to interdependence, adaptation, or influence distribution",
            "what cannot be concluded",
            "human validation checks",
            "stakeholder-facing summary",
        ],
    }


def load_identity_text(identity_path: Optional[str | Path] = None) -> str:
    """Load the chatbot identity file, falling back to a safe default."""

    if identity_path is not None:
        path = Path(identity_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
    return (
        "You are Bio-Behavioral Team Dynamics Analytics Interpreter, a cautious research-support chatbot. "
        "Explain only de-identified or synthetic team dynamics analytics. Help users understand concepts, "
        "metric derivations, visual outputs, and limits of interpretation. Do not evaluate individual trainees, "
        "infer mental states, or claim that entropy, inverse sample entropy, or AMI proves good or bad teamwork."
    )


def build_gemini_prompt(
    identity_text: str,
    packet: Mapping[str, Any],
    user_question: Optional[str] = None,
) -> str:
    """Create a Gemini-ready prompt from identity text, JSON packet, and user question."""

    question = user_question or str(packet.get("user_question", DEFAULT_USER_QUESTION))
    packet_with_question = dict(packet)
    packet_with_question["user_question"] = question
    packet_json = json.dumps(packet_with_question, indent=2)
    return f"""{identity_text}

User question:
{question}

Task:
Answer the user's question as Bio-Behavioral Team Dynamics Analytics Interpreter. The user may understand training but may not know entropy, sample entropy, AMI, or the Fire Support Team context.

Use this structure unless the user asks for something else:
1. Brief answer
2. What the visualization or output shows
3. How the relevant metric is derived in plain language
4. Interpretation connected to interdependence, adaptation, or influence distribution
5. What cannot be concluded
6. Human validation check
7. Stakeholder-facing summary

Constraints:
- Use cautious language.
- Help the user understand both the concepts and the outputs.
- Do not evaluate individual trainees.
- Do not infer stress, workload, cognitive state, or mental state.
- Do not claim causal effects.
- Do not invent data, citations, scenario events, or dissertation findings.
- Make clear that this packet uses synthetic demonstration data.
- Use only information in the identity file and data packet.

Data packet:
{packet_json}
"""


def get_env_api_key() -> Optional[str]:
    """Return a Gemini API key from the environment when configured."""

    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def call_gemini(prompt: str, model: str = "gemini-2.5-flash", api_key: Optional[str] = None) -> str:
    """Generate a chatbot response through the Gemini API.

    The key should be provided through Streamlit secrets or through the
    GEMINI_API_KEY / GOOGLE_API_KEY environment variable. No key should be
    committed to GitHub or stored in a notebook.
    """

    key = api_key or get_env_api_key()
    if not key:
        raise RuntimeError("Gemini API key not configured. Use Streamlit secrets or GEMINI_API_KEY/GOOGLE_API_KEY.")
    try:
        from google import genai  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install google-genai before calling Gemini: pip install google-genai") from exc
    client = genai.Client(api_key=key)
    response = client.models.generate_content(model=model, contents=prompt)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return str(text)


def validate_outputs(
    timeseries_df: pd.DataFrame,
    entropy_df: pd.DataFrame,
    inverse_df: pd.DataFrame,
    ami_long_df: pd.DataFrame,
    ami_summary_df: pd.DataFrame,
    required_files: Sequence[Path],
) -> None:
    """Validate outputs for a clean course/GitHub demonstration."""

    validate_timeseries_dataframe(timeseries_df)
    if entropy_df["entropy_bits"].isna().any() or (entropy_df["entropy_bits"] < 0).any():
        raise ValueError("Entropy values must be nonnegative and nonmissing.")
    if ((entropy_df["normalized_entropy"] < -1e-9) | (entropy_df["normalized_entropy"] > 1.000001)).any():
        raise ValueError("Normalized entropy values should be within [0, 1].")
    if inverse_df["inverse_sample_entropy"].isna().all():
        raise ValueError("All inverse sample entropy values are missing.")
    if ((ami_long_df["ami_share"] < -1e-9) | (ami_long_df["ami_share"] > 1.000001)).any():
        raise ValueError("AMI share values should be within [0, 1].")
    share_sums = ami_long_df.groupby("window_mid_sec")["ami_share"].sum().dropna()
    if not np.allclose(share_sums.values, 1.0, atol=1e-6):
        raise ValueError("AMI shares should sum to 1 within each window.")
    if ami_summary_df.empty:
        raise ValueError("Role AMI summary is empty.")
    missing = [str(path) for path in required_files if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Expected output files were not created: {missing}")


def write_api_status(paths: OutputPaths, api_key_detected: bool, response_generated: bool, model: str) -> None:
    """Write a small status file without including any secret value."""

    status = {
        "api_key_detected": bool(api_key_detected),
        "gemini_response_generated": bool(response_generated),
        "model": model,
        "key_storage_note": "Use Streamlit secrets or environment variables. Do not commit API keys to GitHub.",
        "synthetic_outputs_ready": True,
    }
    paths.api_status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_pipeline(
    output_dir: str | Path = "outputs",
    identity_path: Optional[str | Path] = None,
    seed: int = 42,
    n_seconds: int = 900,
    event_time_sec: Optional[int] = None,
    entropy_window: int = 120,
    sampen_window: int = 180,
    ami_window: int = 180,
    step: int = 15,
    max_lag: int = 5,
    user_question: str = DEFAULT_USER_QUESTION,
    call_api: bool = False,
    model: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
) -> Dict[str, str]:
    """Run the synthetic data, analysis, visualization, and prompt-packet pipeline."""

    paths = build_output_paths(output_dir)
    if event_time_sec is None:
        event_time_sec = n_seconds // 2

    timeseries_df = generate_synthetic_biobehavioral_timeseries(
        n_seconds=n_seconds,
        seed=seed,
        event_time_sec=event_time_sec,
    )
    state_df = extract_symbolic_state_table(timeseries_df)
    entropy_df = moving_window_entropy(timeseries_df, window=entropy_window, step=step)
    inverse_df = moving_window_inverse_sample_entropy(timeseries_df, window=sampen_window, step=step, m=2)
    ami_long_df = moving_window_role_ami(timeseries_df, window=ami_window, step=step, max_lag=max_lag)
    ami_summary_df = summarize_role_ami(ami_long_df)

    timeseries_df.to_csv(paths.synthetic_csv, index=False)
    state_df.to_csv(paths.state_csv, index=False)
    entropy_df.to_csv(paths.entropy_csv, index=False)
    inverse_df.to_csv(paths.inverse_sampen_csv, index=False)
    ami_long_df.to_csv(paths.role_ami_long_csv, index=False)
    ami_summary_df.to_csv(paths.role_ami_summary_csv, index=False)

    plot_team_state(timeseries_df, paths.team_state_figure)
    plot_entropy(entropy_df, paths.entropy_figure)
    plot_inverse_sampen(inverse_df, paths.inverse_sampen_figure)
    plot_role_ami_summary(ami_summary_df, paths.role_ami_figure)
    plot_role_influence_heatmap(ami_long_df, paths.role_influence_heatmap)

    packet = build_explanation_packet(timeseries_df, entropy_df, inverse_df, ami_long_df, ami_summary_df, paths, user_question)
    paths.packet_json.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    prompt = build_gemini_prompt(load_identity_text(identity_path), packet, user_question=user_question)
    paths.prompt_md.write_text(prompt, encoding="utf-8")

    required_files = [
        paths.synthetic_csv,
        paths.state_csv,
        paths.entropy_csv,
        paths.inverse_sampen_csv,
        paths.role_ami_long_csv,
        paths.role_ami_summary_csv,
        paths.team_state_figure,
        paths.entropy_figure,
        paths.inverse_sampen_figure,
        paths.role_ami_figure,
        paths.role_influence_heatmap,
        paths.packet_json,
        paths.prompt_md,
    ]

    response_generated = False
    if call_api:
        response = call_gemini(prompt, model=model, api_key=api_key)
        paths.api_response_md.write_text(response, encoding="utf-8")
        required_files.append(paths.api_response_md)
        response_generated = True
    write_api_status(paths, api_key_detected=bool(api_key or get_env_api_key()), response_generated=response_generated, model=model)
    required_files.append(paths.api_status_json)

    validate_outputs(timeseries_df, entropy_df, inverse_df, ami_long_df, ami_summary_df, required_files)

    outputs = {
        "synthetic_timeseries": str(paths.synthetic_csv),
        "symbolic_states": str(paths.state_csv),
        "entropy_windows": str(paths.entropy_csv),
        "inverse_sample_entropy_windows": str(paths.inverse_sampen_csv),
        "role_ami_long": str(paths.role_ami_long_csv),
        "role_ami_summary": str(paths.role_ami_summary_csv),
        "team_state_figure": str(paths.team_state_figure),
        "entropy_figure": str(paths.entropy_figure),
        "inverse_sample_entropy_figure": str(paths.inverse_sampen_figure),
        "role_ami_summary_figure": str(paths.role_ami_figure),
        "role_influence_heatmap": str(paths.role_influence_heatmap),
        "explanation_packet": str(paths.packet_json),
        "gemini_prompt_payload": str(paths.prompt_md),
        "api_status": str(paths.api_status_json),
    }
    if response_generated:
        outputs["gemini_chatbot_response"] = str(paths.api_response_md)
    return outputs


def main() -> None:
    """CLI entry point used for smoke testing and generating GitHub demo outputs."""

    parser = argparse.ArgumentParser(description="Synthetic bio-behavioral team dynamics chatbot prototype")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--identity-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seconds", type=int, default=900)
    parser.add_argument("--event-time-sec", type=int, default=None)
    parser.add_argument("--entropy-window", type=int, default=120)
    parser.add_argument("--sampen-window", type=int, default=180)
    parser.add_argument("--ami-window", type=int, default=180)
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--call-api", action="store_true")
    args = parser.parse_args()

    outputs = run_pipeline(
        output_dir=args.output_dir,
        identity_path=args.identity_path,
        seed=args.seed,
        n_seconds=args.n_seconds,
        event_time_sec=args.event_time_sec,
        entropy_window=args.entropy_window,
        sampen_window=args.sampen_window,
        ami_window=args.ami_window,
        step=args.step,
        max_lag=args.max_lag,
        call_api=args.call_api,
        model=args.model,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
