"""Streamlit app for the Bio-Behavioral Team Dynamics Analytics Chatbot.

Run from the repository root:
    streamlit run streamlit_app.py

This version keeps the original tab order requested by the user:
    Dashboard -> Synthetic data -> Chatbot

Connection design:
    1. The Synthetic data tab writes the current dataframe to st.session_state.
    2. The analysis bundle is recomputed from that same dataframe.
    3. The Dashboard and Chatbot both read from the same analysis bundle.
    4. Chat responses store figure bytes and table snapshots so the user sees the
       outputs produced from the dataset used for that response.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from team_dynamics_metrics import (  # noqa: E402
    DEFAULT_USER_QUESTION,
    ROLES,
    build_explanation_packet,
    build_gemini_prompt,
    build_output_paths,
    call_gemini,
    extract_symbolic_state_table,
    generate_synthetic_biobehavioral_timeseries,
    load_identity_text,
    moving_window_entropy,
    moving_window_inverse_sample_entropy,
    moving_window_role_ami,
    plot_entropy,
    plot_inverse_sampen,
    plot_role_ami_summary,
    plot_role_influence_heatmap,
    plot_team_state,
    summarize_role_ami,
    validate_outputs,
    write_api_status,
)

IDENTITY_PATH = ROOT / "docs" / "team_dynamics_chatbot_identity.md"
OUTPUT_DIR = ROOT / "outputs"

DEFAULT_SETTINGS: Dict[str, int | str] = {
    "seed": 42,
    "n_seconds": 900,
    "event_time_sec": 450,
    "entropy_window": 120,
    "sampen_window": 180,
    "ami_window": 180,
    "step": 15,
    "model": "gemini-2.5-flash",
}

FIGURE_SPECS = {
    "team_state": {
        "title": "Symbolic team-state trajectory",
        "path_attr": "team_state_figure",
        "caption": "Synthetic symbolic team-state trajectory used by the metric pipeline.",
    },
    "entropy": {
        "title": "Moving-window Shannon entropy",
        "path_attr": "entropy_figure",
        "caption": "Higher values indicate more variability in the symbolic team-state sequence for that window.",
    },
    "inverse_sampen": {
        "title": "Moving-window inverse sample entropy",
        "path_attr": "inverse_sampen_figure",
        "caption": "Higher values indicate more temporal regularity in the symbolic sequence for this synthetic example.",
    },
    "ami_summary": {
        "title": "Role-level AMI summary",
        "path_attr": "role_ami_figure",
        "caption": "Role-level descriptive coupling with the team-state sequence; not an individual evaluation.",
    },
    "ami_heatmap": {
        "title": "Moving-window AMI share heatmap",
        "path_attr": "role_influence_heatmap",
        "caption": "Relative AMI share by role across time windows.",
    },
}

TABLE_SPECS = {
    "entropy": {
        "title": "Moving-window Shannon entropy table",
        "key": "entropy_df",
    },
    "inverse_sampen": {
        "title": "Moving-window inverse sample entropy table",
        "key": "inverse_df",
    },
    "ami_summary": {
        "title": "Role-level AMI summary table",
        "key": "ami_summary_df",
    },
    "ami_long": {
        "title": "Moving-window role AMI table",
        "key": "ami_long_df",
    },
    "synthetic_data": {
        "title": "Synthetic 1 Hz time-series preview",
        "key": "timeseries_df",
    },
    "states": {
        "title": "Symbolic state table",
        "key": "states_df",
    },
}

CONTROL_KEYS = {
    "seed_control": "seed",
    "duration_control": "n_seconds",
    "event_control": "event_time_sec",
    "entropy_window_control": "entropy_window",
    "sampen_window_control": "sampen_window",
    "ami_window_control": "ami_window",
    "step_control": "step",
    "model_control": "model",
}


def get_gemini_key() -> Optional[str]:
    """Read Gemini key from Streamlit secrets, then environment variables."""

    secret_key = None
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        secret_key = None
    return secret_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def app_settings_from_session() -> Dict[str, int | str]:
    """Return the active settings that define the current dataset and analyses."""

    settings = dict(DEFAULT_SETTINGS)
    settings.update(st.session_state.get("settings", {}))
    return settings


def control_settings_from_widgets() -> Dict[str, int | str]:
    """Read pending settings from Synthetic data tab widgets."""

    settings = dict(DEFAULT_SETTINGS)
    for widget_key, setting_key in CONTROL_KEYS.items():
        if widget_key in st.session_state:
            settings[setting_key] = st.session_state[widget_key]
        else:
            settings[setting_key] = st.session_state.get("settings", DEFAULT_SETTINGS).get(setting_key, DEFAULT_SETTINGS[setting_key])
    return settings


def sync_control_widgets(settings: Dict[str, int | str]) -> None:
    """Synchronize widget defaults after dataset changes from the chatbot."""

    for widget_key, setting_key in CONTROL_KEYS.items():
        st.session_state[widget_key] = settings[setting_key]
    st.session_state.controls_need_sync = False


def dataset_id_from_dataframe(df: pd.DataFrame, settings: Dict[str, int | str]) -> str:
    """Create a compact fingerprint for the current synthetic data and settings."""

    hasher = hashlib.sha1()
    hasher.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return hasher.hexdigest()[:12]


def reset_chat_with_note(note: str) -> None:
    """Reset chat history when the user intentionally switches datasets."""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": note,
            "artifacts": {},
            "dataset_id": st.session_state.get("dataset_id", "not_initialized"),
        }
    ]




def clear_generation_widget_state() -> None:
    """Clear Synthetic data widget values so they can be resynced after chatbot-driven generation."""

    for widget_key in list(CONTROL_KEYS):
        if widget_key in st.session_state:
            del st.session_state[widget_key]


def run_analysis_for_dataframe(
    df: pd.DataFrame,
    *,
    entropy_window: int,
    sampen_window: int,
    ami_window: int,
    step: int,
    model: str,
    user_question: str = DEFAULT_USER_QUESTION,
    settings_for_id: Optional[Dict[str, int | str]] = None,
) -> Dict[str, Any]:
    """Run all local Python analyses on the current dataframe.

    The returned bundle is the single source of truth used by Dashboard and
    Chatbot. Files are also written for GitHub reproducibility, but the live UI
    reads the in-memory bundle and figure bytes rather than relying on a static
    files tab.
    """

    paths = build_output_paths(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    active_settings = dict(app_settings_from_session())
    if settings_for_id:
        active_settings.update(settings_for_id)
    active_settings.update(
        {
            "entropy_window": int(entropy_window),
            "sampen_window": int(sampen_window),
            "ami_window": int(ami_window),
            "step": int(step),
            "model": str(model),
        }
    )
    current_dataset_id = dataset_id_from_dataframe(df, active_settings)

    states_df = extract_symbolic_state_table(df)
    entropy_df = moving_window_entropy(df, window=int(entropy_window), step=int(step))
    inverse_df = moving_window_inverse_sample_entropy(df, window=int(sampen_window), step=int(step))
    ami_long_df = moving_window_role_ami(df, window=int(ami_window), step=int(step))
    ami_summary_df = summarize_role_ami(ami_long_df)

    df.to_csv(paths.synthetic_csv, index=False)
    states_df.to_csv(paths.state_csv, index=False)
    entropy_df.to_csv(paths.entropy_csv, index=False)
    inverse_df.to_csv(paths.inverse_sampen_csv, index=False)
    ami_long_df.to_csv(paths.role_ami_long_csv, index=False)
    ami_summary_df.to_csv(paths.role_ami_summary_csv, index=False)

    plot_team_state(df, paths.team_state_figure)
    plot_entropy(entropy_df, paths.entropy_figure)
    plot_inverse_sampen(inverse_df, paths.inverse_sampen_figure)
    plot_role_ami_summary(ami_summary_df, paths.role_ami_figure)
    plot_role_influence_heatmap(ami_long_df, paths.role_influence_heatmap)

    packet = build_explanation_packet(
        df,
        entropy_df,
        inverse_df,
        ami_long_df,
        ami_summary_df,
        paths,
        user_question=user_question,
    )
    packet["streamlit_connection_status"] = {
        "dataset_id": current_dataset_id,
        "uses_current_synthetic_data_from_session_state": True,
        "local_python_runs_metrics_before_llm_explanation": True,
        "dashboard_and_chatbot_share_same_analysis_bundle": True,
        "settings": active_settings,
    }
    paths.packet_json.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    write_api_status(paths, api_key_detected=bool(get_gemini_key()), response_generated=False, model=str(model))

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
        paths.api_status_json,
    ]
    validate_outputs(df, entropy_df, inverse_df, ami_long_df, ami_summary_df, required_files)

    return {
        "paths": paths,
        "timeseries_df": df,
        "states_df": states_df,
        "entropy_df": entropy_df,
        "inverse_df": inverse_df,
        "ami_long_df": ami_long_df,
        "ami_summary_df": ami_summary_df,
        "packet": packet,
        "dataset_id": current_dataset_id,
        "last_analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": active_settings,
    }


def generate_data_and_run_analysis(
    settings: Dict[str, int | str],
    *,
    user_question: str = DEFAULT_USER_QUESTION,
) -> None:
    """Generate a connected synthetic dataset and immediately run analyses."""

    df = generate_synthetic_biobehavioral_timeseries(
        n_seconds=int(settings["n_seconds"]),
        seed=int(settings["seed"]),
        event_time_sec=int(settings["event_time_sec"]),
    )
    bundle = run_analysis_for_dataframe(
        df,
        entropy_window=int(settings["entropy_window"]),
        sampen_window=int(settings["sampen_window"]),
        ami_window=int(settings["ami_window"]),
        step=int(settings["step"]),
        model=str(settings["model"]),
        user_question=user_question,
        settings_for_id=settings,
    )
    previous_dataset_id = st.session_state.get("dataset_id")
    st.session_state.timeseries_df = df
    st.session_state.analysis_bundle = bundle
    st.session_state.settings = dict(settings)
    st.session_state.dataset_id = bundle["dataset_id"]
    if previous_dataset_id and previous_dataset_id != bundle["dataset_id"]:
        st.session_state.dataset_version = int(st.session_state.get("dataset_version", 1)) + 1
    else:
        st.session_state.dataset_version = int(st.session_state.get("dataset_version", 1))


def rerun_analysis_on_current_dataframe(
    settings: Dict[str, int | str],
    *,
    user_question: str = DEFAULT_USER_QUESTION,
) -> None:
    """Keep the current synthetic data but recompute analyses with current windows."""

    if "timeseries_df" not in st.session_state:
        generate_data_and_run_analysis(settings, user_question=user_question)
        return
    bundle = run_analysis_for_dataframe(
        st.session_state.timeseries_df,
        entropy_window=int(settings["entropy_window"]),
        sampen_window=int(settings["sampen_window"]),
        ami_window=int(settings["ami_window"]),
        step=int(settings["step"]),
        model=str(settings["model"]),
        user_question=user_question,
        settings_for_id=settings,
    )
    st.session_state.analysis_bundle = bundle
    st.session_state.settings = dict(settings)
    st.session_state.dataset_id = bundle["dataset_id"]


def ensure_connected_state() -> None:
    """Initialize synthetic data, analysis outputs, and chat state."""

    if "settings" not in st.session_state:
        st.session_state.settings = dict(DEFAULT_SETTINGS)
    if "dataset_version" not in st.session_state:
        st.session_state.dataset_version = 1
    if "timeseries_df" not in st.session_state or "analysis_bundle" not in st.session_state:
        generate_data_and_run_analysis(app_settings_from_session())
        st.session_state.controls_need_sync = True
    if "messages" not in st.session_state:
        reset_chat_with_note(
            "I am connected to the current synthetic dataset. Change the dataset in the Synthetic data tab, "
            "then ask me to run analyses, show figures or tables, or explain the outputs."
        )


def parse_generation_overrides(question: str, settings: Dict[str, int | str]) -> Dict[str, int | str]:
    """Parse simple generation overrides from a user message."""

    updated = dict(settings)
    lower = question.lower()
    seed_match = re.search(r"seed\s*(?:=|is|:)?\s*(\d+)", lower)
    seconds_match = re.search(r"(\d{3,4})\s*(?:seconds|sec|s)\b", lower)
    duration_match = re.search(r"duration\s*(?:=|is|:)?\s*(\d{3,4})", lower)
    event_match = re.search(r"event\s*(?:time|at|=|is|:)?\s*(\d{2,4})", lower)

    if seed_match:
        updated["seed"] = int(seed_match.group(1))
    if duration_match:
        updated["n_seconds"] = max(300, min(1800, int(duration_match.group(1))))
    elif seconds_match:
        updated["n_seconds"] = max(300, min(1800, int(seconds_match.group(1))))
    if event_match:
        event_time = int(event_match.group(1))
        n_seconds = int(updated["n_seconds"])
        updated["event_time_sec"] = max(30, min(n_seconds - 30, event_time))
    return updated


def classify_user_request(question: str) -> Dict[str, Any]:
    """Classify the user request into local analysis actions and UI artifacts."""

    q = question.lower()
    wants_generate = any(
        term in q
        for term in [
            "generate",
            "create",
            "new synthetic",
            "new data",
            "new dataset",
            "regenerate",
            "simulate",
            "different data",
            "different dataset",
            "change data",
            "change the data",
            "use seed",
            "set seed",
            "change seed",
            "with seed",
        ]
    )
    wants_run = any(term in q for term in ["run", "compute", "calculate", "analyze", "analyse", "analysis", "analyses"])
    wants_figures = any(term in q for term in ["figure", "figures", "plot", "plots", "graph", "visual", "visualization"])
    wants_tables = any(term in q for term in ["table", "tables", "dataframe", "data frame", "csv", "values", "output"])

    figure_keys: List[str] = []
    table_keys: List[str] = []

    if any(term in q for term in ["entropy", "shannon", "adaptation", "reorganization", "reorganisation"]):
        figure_keys.append("entropy")
        table_keys.append("entropy")
    if any(term in q for term in ["sample", "sampen", "inverse", "regularity", "interdependence"]):
        figure_keys.append("inverse_sampen")
        table_keys.append("inverse_sampen")
    if any(term in q for term in ["ami", "mutual information", "influence", "role", "jtac", "fso", "fom", "foa", "leader"]):
        figure_keys.extend(["ami_summary", "ami_heatmap"])
        table_keys.extend(["ami_summary", "ami_long"])
    if any(term in q for term in ["team state", "symbolic state", "state trajectory", "timeline"]):
        figure_keys.append("team_state")
        table_keys.append("states")

    asks_for_everything = wants_run and (
        "all" in q or "everything" in q or "figures and tables" in q or "outputs" in q or "data analyses" in q
    )
    if asks_for_everything or (wants_figures and not figure_keys):
        figure_keys.extend(["team_state", "entropy", "inverse_sampen", "ami_summary", "ami_heatmap"])
    if asks_for_everything or (wants_tables and not table_keys):
        table_keys.extend(["entropy", "inverse_sampen", "ami_summary", "ami_long"])

    figure_keys = list(dict.fromkeys(figure_keys))
    table_keys = list(dict.fromkeys(table_keys))

    return {
        "generate": wants_generate,
        "run_analysis": wants_run or wants_generate,
        "figure_keys": figure_keys,
        "table_keys": table_keys,
        "wants_gemini_explanation": not (wants_figures or wants_tables) or any(
            term in q for term in ["explain", "interpret", "meaning", "understand", "why", "what does", "what do"]
        ),
    }


def local_analysis_summary(bundle: Dict[str, Any]) -> str:
    """Create a factual summary of the current Python-computed outputs."""

    packet = bundle["packet"]
    entropy = packet["adaptation_entropy_summary"]
    inverse = packet["interdependence_inverse_sample_entropy_summary"]
    influence = packet["influence_distribution_summary"]
    settings = bundle.get("settings", app_settings_from_session())
    dataset_id = bundle.get("dataset_id", st.session_state.get("dataset_id", "unknown"))
    n_rows = len(bundle["timeseries_df"])
    return (
        f"I ran the local Python analyses on the current synthetic 1 Hz dataset "
        f"(dataset ID `{dataset_id}`, {n_rows:,} rows, seed={settings['seed']}, "
        f"synthetic event={settings['event_time_sec']} sec).\n\n"
        f"**Computed outputs:** Shannon entropy mean = {entropy['mean_entropy_bits']}, "
        f"peak = {entropy['peak_entropy_bits']} bits at {entropy['peak_time_min']} min; "
        f"inverse sample entropy mean = {inverse['mean_inverse_sample_entropy']}, "
        f"peak = {inverse['peak_inverse_sample_entropy']} at {inverse['peak_time_min']} min; "
        f"top role by mean AMI share = {influence['top_role_by_mean_ami_share']} "
        f"({influence['top_role_mean_ami_share']}).\n\n"
        "These values are synthetic demonstration outputs. They show that the pipeline is connected and running; "
        "they are not dissertation findings and should not be interpreted as real trainee performance."
    )


def build_chat_prompt_with_current_outputs(
    question: str,
    bundle: Dict[str, Any],
    requested_figures: List[str],
    requested_tables: List[str],
) -> str:
    """Build a Gemini prompt that separates local computation from LLM explanation."""

    identity_text = load_identity_text(IDENTITY_PATH)
    packet = dict(bundle["packet"])
    packet["user_question"] = question
    packet["local_python_status"] = {
        "analysis_has_been_run": True,
        "analysis_source": "Streamlit app local Python functions",
        "llm_role": "Explain the already-computed outputs; do not invent additional calculations.",
        "requested_figures": requested_figures,
        "requested_tables": requested_tables,
    }
    base_prompt = build_gemini_prompt(identity_text, packet, user_question=question)
    extra = """

Additional instruction for this Streamlit version:
The user is interacting with a connected app. Python has already generated the current synthetic data and computed the metric tables and figures. You should explain the current outputs and, when relevant, refer to the figures/tables that the app displays below your answer. Do not claim that Gemini ran Python code, do not invent values not in the JSON packet, and do not evaluate individual trainees.
"""
    return base_prompt + extra


def figure_payloads_for_message(figure_keys: List[str], bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Store figure bytes in the chat message to avoid stale cached files."""

    payloads: List[Dict[str, Any]] = []
    paths = bundle["paths"]
    for key in figure_keys:
        spec = FIGURE_SPECS[key]
        path = getattr(paths, spec["path_attr"])
        if path.exists():
            payloads.append(
                {
                    "key": key,
                    "title": spec["title"],
                    "caption": f"{spec['caption']} Dataset ID: {bundle['dataset_id']}.",
                    "image_bytes": path.read_bytes(),
                }
            )
    return payloads


def table_payloads_for_message(table_keys: List[str], bundle: Dict[str, Any], *, max_rows: int = 80) -> List[Dict[str, Any]]:
    """Store table snapshots in the chat message."""

    payloads: List[Dict[str, Any]] = []
    for key in table_keys:
        spec = TABLE_SPECS[key]
        table_df = bundle[spec["key"]].head(max_rows).copy()
        payloads.append(
            {
                "key": key,
                "title": f"{spec['title']} (dataset ID {bundle['dataset_id']})",
                "df": table_df,
            }
        )
    return payloads


def artifact_payload_for_message(
    figure_keys: List[str],
    table_keys: List[str],
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Create immutable chat artifacts from the current analysis bundle."""

    return {
        "dataset_id": bundle["dataset_id"],
        "figures": figure_payloads_for_message(figure_keys, bundle),
        "tables": table_payloads_for_message(table_keys, bundle),
    }


def render_metric_cards(bundle: Dict[str, Any]) -> None:
    """Display compact metric cards for the current analysis bundle."""

    packet = bundle["packet"]
    entropy = packet["adaptation_entropy_summary"]
    inverse = packet["interdependence_inverse_sample_entropy_summary"]
    influence = packet["influence_distribution_summary"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dataset ID", bundle["dataset_id"])
    c2.metric("Rows", f"{len(bundle['timeseries_df']):,}")
    c3.metric("Peak entropy", f"{entropy['peak_entropy_bits']:.3f} bits", f"{entropy['peak_time_min']:.2f} min")
    c4.metric("Peak inverse SampEn", f"{inverse['peak_inverse_sample_entropy']:.3f}", f"{inverse['peak_time_min']:.2f} min")
    c5.metric("Top mean AMI share", influence["top_role_by_mean_ami_share"], f"{influence['top_role_mean_ami_share']:.3f}")


def render_figures_from_bundle(figure_keys: List[str], bundle: Dict[str, Any]) -> None:
    """Render current dashboard figures from bytes rather than static browser-cached paths."""

    paths = bundle["paths"]
    for key in figure_keys:
        spec = FIGURE_SPECS[key]
        path = getattr(paths, spec["path_attr"])
        if path.exists():
            st.image(path.read_bytes(), caption=f"{spec['caption']} Dataset ID: {bundle['dataset_id']}.", use_container_width=True)
        else:
            st.warning(f"Missing figure: {path.name}")


def render_tables_from_bundle(table_keys: List[str], bundle: Dict[str, Any], *, max_rows: int = 80) -> None:
    """Render current dashboard tables."""

    for key in table_keys:
        spec = TABLE_SPECS[key]
        df = bundle[spec["key"]]
        st.markdown(f"**{spec['title']}**")
        st.dataframe(df.head(max_rows), use_container_width=True)


def render_message(message: Dict[str, Any]) -> None:
    """Render a stored chat message with its own figure/table snapshots."""

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        artifacts = message.get("artifacts", {}) or {}
        for fig in artifacts.get("figures", []):
            st.image(fig["image_bytes"], caption=fig["caption"], use_container_width=True)
        for table in artifacts.get("tables", []):
            st.markdown(f"**{table['title']}**")
            st.dataframe(table["df"], use_container_width=True)


def current_dataset_status(bundle: Dict[str, Any]) -> str:
    """Return a compact status line for the active connected dataset."""

    settings = bundle.get("settings", app_settings_from_session())
    return (
        f"Connected dataset ID `{bundle['dataset_id']}` | version {st.session_state.get('dataset_version', 1)} | "
        f"{len(bundle['timeseries_df']):,} rows at 1 Hz | seed={settings['seed']} | "
        f"event={settings['event_time_sec']} sec | last analysis={bundle['last_analysis_time']}"
    )


st.set_page_config(
    page_title="Bio-Behavioral Team Dynamics Chatbot",
    page_icon="💬",
    layout="wide",
)

ensure_connected_state()
if st.session_state.get("controls_need_sync", False):
    sync_control_widgets(app_settings_from_session())

bundle = st.session_state.analysis_bundle
settings = app_settings_from_session()

st.title("Bio-Behavioral Team Dynamics Analytics Chatbot Prototype")
st.caption(
    "Synthetic demonstration data only. The Dashboard, Synthetic data tab, and Chatbot now share the same "
    "session-state dataset and analysis bundle."
)

with st.sidebar:
    st.header("Connection status")
    st.write(current_dataset_status(bundle))
    st.divider()
    st.header("Gemini API")
    api_key = get_gemini_key()
    if api_key:
        st.success("Gemini key detected.")
    else:
        st.warning("No Gemini key detected.")
        st.caption("Use `.streamlit/secrets.toml` locally or Streamlit Cloud secrets. Do not paste keys into the notebook.")
    st.divider()
    st.header("Suggested chatbot prompts")
    st.markdown(
        "- Run all analyses and show figures and tables.\n"
        "- Explain the entropy peak using the current dataset.\n"
        "- Show the AMI table for the current synthetic data.\n"
        "- Generate a new synthetic dataset with seed 7, 1200 seconds, event 600, then run all analyses."
    )

st.info(current_dataset_status(bundle))

tab_dashboard, tab_data, tab_chat = st.tabs(["Dashboard", "Synthetic data", "Chatbot"])

with tab_dashboard:
    st.subheader("Dashboard for the current synthetic dataset")
    st.write(
        "This dashboard reads the same analysis bundle that the chatbot uses. To change the dataset, go to the "
        "Synthetic data tab, generate a new dataset, and return here. The dataset ID will change when the data change."
    )
    render_metric_cards(bundle)
    c1, c2 = st.columns(2)
    with c1:
        render_figures_from_bundle(["team_state", "entropy", "ami_summary"], bundle)
    with c2:
        render_figures_from_bundle(["inverse_sampen", "ami_heatmap"], bundle)

    st.subheader("Metric tables")
    table_choice = st.selectbox(
        "Choose a table",
        options=["entropy", "inverse_sampen", "ami_summary", "ami_long", "synthetic_data", "states"],
        format_func=lambda key: TABLE_SPECS[key]["title"],
        key="dashboard_table_choice",
    )
    render_tables_from_bundle([table_choice], bundle, max_rows=150)

with tab_data:
    st.subheader("Synthetic data generator")
    st.write(
        "Use this tab to create the dataset used by both the dashboard and chatbot. The app generates a synthetic "
        "1 Hz time series with placeholder role-level bio-behavioral features, symbolic role states, phase labels, "
        "a symbolic team state, and one synthetic event marker."
    )

    st.markdown("#### Dataset settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Random seed", min_value=1, max_value=9999, value=int(settings["seed"]), step=1, key="seed_control")
    with c2:
        st.slider(
            "Duration (seconds, 1 Hz)",
            min_value=300,
            max_value=1800,
            value=int(settings["n_seconds"]),
            step=60,
            key="duration_control",
        )
    with c3:
        max_event = max(60, int(st.session_state.get("duration_control", settings["n_seconds"])) - 30)
        current_event_value = min(int(settings["event_time_sec"]), max_event)
        st.slider(
            "Synthetic event time (seconds)",
            min_value=30,
            max_value=max_event,
            value=current_event_value,
            step=15,
            key="event_control",
        )

    st.markdown("#### Analysis settings")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        st.slider("Entropy window (seconds)", min_value=60, max_value=300, value=int(settings["entropy_window"]), step=15, key="entropy_window_control")
    with c5:
        st.slider("Sample entropy window (seconds)", min_value=90, max_value=360, value=int(settings["sampen_window"]), step=15, key="sampen_window_control")
    with c6:
        st.slider("AMI window (seconds)", min_value=90, max_value=360, value=int(settings["ami_window"]), step=15, key="ami_window_control")
    with c7:
        st.slider("Window step (seconds)", min_value=5, max_value=60, value=int(settings["step"]), step=5, key="step_control")
    st.text_input("Gemini model", value=str(settings["model"]), key="model_control")

    pending_settings = control_settings_from_widgets()
    settings_changed = pending_settings != settings
    if settings_changed:
        st.warning("You have pending settings changes. Click one of the buttons below to apply them to the dashboard and chatbot.")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Generate synthetic data + run analyses", type="primary", use_container_width=True):
            with st.spinner("Generating a new synthetic time series and recomputing all outputs..."):
                generate_data_and_run_analysis(pending_settings)
            reset_chat_with_note(
                "The synthetic dataset was regenerated, and I am now connected to the new data. "
                "Ask me to run analyses, show figures or tables, or explain the outputs."
            )
            st.success("Synthetic data, dashboard, and chatbot were updated together.")
            st.rerun()
    with b2:
        if st.button("Keep current data + rerun analyses", use_container_width=True):
            with st.spinner("Recomputing analyses on the current synthetic time series..."):
                rerun_analysis_on_current_dataframe(pending_settings)
            reset_chat_with_note(
                "The existing synthetic dataset was kept, and the analyses were recomputed with the current window settings."
            )
            st.success("Dashboard and chatbot analyses were updated together.")
            st.rerun()

    st.markdown("#### Current connected synthetic data")
    render_metric_cards(bundle)
    st.dataframe(bundle["timeseries_df"].head(100), use_container_width=True)
    with st.expander("Symbolic state table used for entropy, sample entropy, and AMI"):
        st.dataframe(bundle["states_df"].head(100), use_container_width=True)
    with st.expander("Download current connected outputs"):
        st.download_button(
            "Download current synthetic time series CSV",
            data=bundle["timeseries_df"].to_csv(index=False).encode("utf-8"),
            file_name=f"synthetic_biobehavioral_timeseries_{bundle['dataset_id']}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download current explanation packet JSON",
            data=json.dumps(bundle["packet"], indent=2).encode("utf-8"),
            file_name=f"team_dynamics_explanation_packet_{bundle['dataset_id']}.json",
            mime="application/json",
        )

with tab_chat:
    st.subheader("Chatbot connected to the current synthetic data and analyses")
    st.write(
        "The chatbot uses the current dataset ID shown below. When you ask it to run analyses, Python recomputes "
        "the metrics from the active synthetic dataframe. If Gemini is configured, Gemini explains those already-computed outputs."
    )
    st.success(current_dataset_status(bundle))

    for message in st.session_state.messages:
        render_message(message)

    user_question = st.chat_input("Ask about the current synthetic data, run analyses, or request figures/tables")
    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
                "artifacts": {},
                "dataset_id": st.session_state.get("dataset_id"),
            }
        )
        request = classify_user_request(user_question)
        settings = app_settings_from_session()

        with st.spinner("Processing request with the connected Python analysis pipeline..."):
            if request["generate"]:
                updated_settings = parse_generation_overrides(user_question, settings)
                generate_data_and_run_analysis(updated_settings, user_question=user_question)
                st.session_state.controls_need_sync = True
                bundle = st.session_state.analysis_bundle
                settings = app_settings_from_session()
                clear_generation_widget_state()
            elif request["run_analysis"]:
                rerun_analysis_on_current_dataframe(settings, user_question=user_question)
                bundle = st.session_state.analysis_bundle
            else:
                bundle = st.session_state.analysis_bundle

        figure_keys = request["figure_keys"]
        table_keys = request["table_keys"]
        if request["run_analysis"] and not figure_keys and not table_keys:
            figure_keys = ["team_state", "entropy", "inverse_sampen", "ami_summary", "ami_heatmap"]
            table_keys = ["entropy", "inverse_sampen", "ami_summary"]

        prompt = build_chat_prompt_with_current_outputs(user_question, bundle, figure_keys, table_keys)
        bundle["paths"].prompt_md.write_text(prompt, encoding="utf-8")

        local_summary = local_analysis_summary(bundle)
        if get_gemini_key() and request["wants_gemini_explanation"]:
            with st.spinner("Calling Gemini to explain the computed outputs..."):
                try:
                    response = call_gemini(prompt, model=str(settings["model"]), api_key=get_gemini_key())
                    bundle["paths"].api_response_md.write_text(response, encoding="utf-8")
                    write_api_status(bundle["paths"], api_key_detected=True, response_generated=True, model=str(settings["model"]))
                except Exception as exc:
                    response = local_summary + f"\n\nGemini explanation failed: `{exc}`"
        else:
            response = local_summary
            if not get_gemini_key():
                response += "\n\nGemini is not configured, so this response is the local Python-computed summary plus displayed artifacts."
            elif not request["wants_gemini_explanation"]:
                response += "\n\nI displayed the requested computed artifacts without asking Gemini for a narrative explanation."

        artifacts = artifact_payload_for_message(figure_keys, table_keys, bundle)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "artifacts": artifacts,
                "dataset_id": bundle["dataset_id"],
            }
        )
        st.rerun()

