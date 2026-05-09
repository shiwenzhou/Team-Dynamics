"""Connected Streamlit app for the Bio-Behavioral Team Dynamics Analytics Chatbot.

Run from the repository root:
    streamlit run streamlit_app.py

What changed in this version:
    - The synthetic data, dashboard, and chatbot all use the same in-memory session state.
    - The chatbot can trigger the Python analysis pipeline and display current figures/tables.
    - Gemini only explains already-computed outputs; Python performs the calculations.
    - API keys are read from Streamlit secrets or environment variables, never from the notebook.
"""

from __future__ import annotations

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
}


def get_gemini_key() -> Optional[str]:
    """Read Gemini key from Streamlit secrets, then from environment variables."""

    secret_key = None
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        secret_key = None
    return secret_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def app_settings_from_session() -> Dict[str, int | str]:
    """Return the current analysis settings from session state."""

    settings = dict(DEFAULT_SETTINGS)
    settings.update(st.session_state.get("settings", {}))
    return settings


def reset_chat_with_note(note: str) -> None:
    """Reset chat history when the connected dataset changes."""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": note,
            "artifacts": [],
        }
    ]


def run_analysis_for_dataframe(
    df: pd.DataFrame,
    *,
    entropy_window: int,
    sampen_window: int,
    ami_window: int,
    step: int,
    model: str,
    user_question: str = DEFAULT_USER_QUESTION,
) -> Dict[str, Any]:
    """Run all analyses on the current synthetic dataframe and save public outputs.

    The returned bundle is the single source of truth used by the dashboard and
    chatbot. The app writes CSV/PNG/JSON outputs for GitHub reproducibility, but
    the UI does not rely on a separate 'Generated files' page.
    """

    paths = build_output_paths(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        "last_analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "entropy_window": int(entropy_window),
            "sampen_window": int(sampen_window),
            "ami_window": int(ami_window),
            "step": int(step),
            "model": str(model),
        },
    }


def generate_data_and_run_analysis(settings: Dict[str, int | str], *, user_question: str = DEFAULT_USER_QUESTION) -> None:
    """Generate the connected synthetic dataset and immediately run analyses."""

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
    )
    st.session_state.timeseries_df = df
    st.session_state.analysis_bundle = bundle
    st.session_state.settings = dict(settings)


def ensure_connected_state() -> None:
    """Initialize synthetic data, analysis outputs, and chat state."""

    if "settings" not in st.session_state:
        st.session_state.settings = dict(DEFAULT_SETTINGS)
    if "timeseries_df" not in st.session_state or "analysis_bundle" not in st.session_state:
        generate_data_and_run_analysis(app_settings_from_session())
    if "messages" not in st.session_state:
        reset_chat_with_note(
            "I am connected to the current synthetic dataset. You can ask me to run the analyses, "
            "show entropy, show inverse sample entropy, show AMI, display tables, or explain a figure."
        )


def parse_generation_overrides(question: str, settings: Dict[str, int | str]) -> Dict[str, int | str]:
    """Parse simple generation overrides from a user message."""

    updated = dict(settings)
    lower = question.lower()
    seed_match = re.search(r"seed\s*(?:=|is|:)?\s*(\d+)", lower)
    seconds_match = re.search(r"(\d{3,4})\s*(?:seconds|sec|s)\b", lower)
    event_match = re.search(r"event\s*(?:time|at|=|is|:)?\s*(\d{2,4})", lower)

    if seed_match:
        updated["seed"] = int(seed_match.group(1))
    if seconds_match:
        updated["n_seconds"] = max(300, min(1800, int(seconds_match.group(1))))
    if event_match:
        event_time = int(event_match.group(1))
        n_seconds = int(updated["n_seconds"])
        updated["event_time_sec"] = max(30, min(n_seconds - 30, event_time))
    return updated


def classify_user_request(question: str) -> Dict[str, Any]:
    """Classify the user's request into local analysis actions and UI artifacts."""

    q = question.lower()
    wants_generate = any(term in q for term in ["generate", "create", "new synthetic", "regenerate", "simulate"])
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
        table_keys.append("synthetic_data")

    asks_for_everything = wants_run and (
        "all" in q or "everything" in q or "figures and tables" in q or "outputs" in q or "data analyses" in q
    )
    if asks_for_everything or (wants_figures and not figure_keys):
        figure_keys.extend(["team_state", "entropy", "inverse_sampen", "ami_summary", "ami_heatmap"])
    if asks_for_everything or (wants_tables and not table_keys):
        table_keys.extend(["entropy", "inverse_sampen", "ami_summary"])

    # Remove duplicates while preserving order.
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
    n_rows = len(bundle["timeseries_df"])
    settings = app_settings_from_session()
    return (
        f"I ran the local Python analyses on the current synthetic 1 Hz dataset ({n_rows:,} rows; "
        f"seed={settings['seed']}; synthetic event={settings['event_time_sec']} sec).\n\n"
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
    """Build a Gemini prompt that clearly separates local computation from LLM explanation."""

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
The user is interacting with a connected app. Python has already generated the synthetic data and computed the metric tables and figures. You should explain the current outputs and, when relevant, refer to the figures/tables that the app displays below your answer. Do not claim that Gemini ran Python code, do not invent values not in the JSON packet, and do not evaluate individual trainees.
"""
    return base_prompt + extra


def render_metric_cards(bundle: Dict[str, Any]) -> None:
    """Display compact metric cards for the current analysis bundle."""

    packet = bundle["packet"]
    entropy = packet["adaptation_entropy_summary"]
    inverse = packet["interdependence_inverse_sample_entropy_summary"]
    influence = packet["influence_distribution_summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(bundle['timeseries_df']):,}")
    c2.metric("Peak entropy", f"{entropy['peak_entropy_bits']:.3f} bits", f"{entropy['peak_time_min']:.2f} min")
    c3.metric("Peak inverse SampEn", f"{inverse['peak_inverse_sample_entropy']:.3f}", f"{inverse['peak_time_min']:.2f} min")
    c4.metric("Top mean AMI share", influence["top_role_by_mean_ami_share"], f"{influence['top_role_mean_ami_share']:.3f}")


def render_figures(figure_keys: List[str], bundle: Dict[str, Any]) -> None:
    """Render figure artifacts for a chat response or dashboard section."""

    paths = bundle["paths"]
    for key in figure_keys:
        spec = FIGURE_SPECS[key]
        path = getattr(paths, spec["path_attr"])
        if path.exists():
            st.image(str(path), caption=spec["caption"], use_container_width=True)
        else:
            st.warning(f"Missing figure: {path.name}")


def render_tables(table_keys: List[str], bundle: Dict[str, Any], *, max_rows: int = 80) -> None:
    """Render table artifacts for a chat response or dashboard section."""

    for key in table_keys:
        spec = TABLE_SPECS[key]
        df = bundle[spec["key"]]
        st.markdown(f"**{spec['title']}**")
        st.dataframe(df.head(max_rows), use_container_width=True)


def render_message(message: Dict[str, Any], bundle: Dict[str, Any]) -> None:
    """Render a stored chat message with optional tables and figures."""

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        artifacts = message.get("artifacts", {}) or {}
        figure_keys = artifacts.get("figures", [])
        table_keys = artifacts.get("tables", [])
        if figure_keys:
            render_figures(figure_keys, bundle)
        if table_keys:
            render_tables(table_keys, bundle)


st.set_page_config(
    page_title="Bio-Behavioral Team Dynamics Chatbot",
    page_icon="💬",
    layout="wide",
)

ensure_connected_state()
bundle = st.session_state.analysis_bundle
settings = app_settings_from_session()

st.title("Bio-Behavioral Team Dynamics Analytics Chatbot Prototype")
st.caption(
    "Synthetic demonstration data only. The Streamlit app connects synthetic data generation, metric analyses, "
    "dashboard outputs, and chatbot interpretation in one workflow."
)

with st.sidebar:
    st.header("Current synthetic dataset")
    st.caption("Changing these settings and clicking the button regenerates the dataset and reruns all analyses.")
    settings["seed"] = st.number_input("Random seed", min_value=1, max_value=9999, value=int(settings["seed"]), step=1)
    settings["n_seconds"] = st.slider(
        "Duration (seconds, 1 Hz)", min_value=300, max_value=1800, value=int(settings["n_seconds"]), step=60
    )
    settings["event_time_sec"] = st.slider(
        "Synthetic event time (seconds)",
        min_value=30,
        max_value=int(settings["n_seconds"]) - 30,
        value=min(int(settings["event_time_sec"]), int(settings["n_seconds"]) - 30),
        step=15,
    )
    st.divider()
    st.header("Analysis settings")
    settings["entropy_window"] = st.slider(
        "Entropy window (seconds)", min_value=60, max_value=300, value=int(settings["entropy_window"]), step=15
    )
    settings["sampen_window"] = st.slider(
        "Sample entropy window (seconds)", min_value=90, max_value=360, value=int(settings["sampen_window"]), step=15
    )
    settings["ami_window"] = st.slider(
        "AMI window (seconds)", min_value=90, max_value=360, value=int(settings["ami_window"]), step=15
    )
    settings["step"] = st.slider("Window step (seconds)", min_value=5, max_value=60, value=int(settings["step"]), step=5)
    settings["model"] = st.text_input("Gemini model", value=str(settings["model"]))

    if st.button("Generate synthetic data + run analyses", type="primary", use_container_width=True):
        with st.spinner("Generating synthetic time series and computing all metric outputs..."):
            generate_data_and_run_analysis(settings)
        reset_chat_with_note(
            "I regenerated the synthetic dataset and reran the analyses. Ask me to show figures/tables or explain the outputs."
        )
        st.rerun()

    st.divider()
    st.header("Gemini API")
    api_key = get_gemini_key()
    if api_key:
        st.success("Gemini key detected.")
    else:
        st.warning("No Gemini key detected.")
        st.caption("Use `.streamlit/secrets.toml` locally or Streamlit Cloud secrets. Do not paste keys into the notebook.")

bundle = st.session_state.analysis_bundle
settings = app_settings_from_session()

st.info(
    f"Connected dataset: {len(bundle['timeseries_df']):,} synthetic 1 Hz rows; seed={settings['seed']}; "
    f"synthetic event={settings['event_time_sec']} sec; last analysis={bundle['last_analysis_time']}."
)

tab_data, tab_dashboard, tab_chat, tab_equations = st.tabs(
    ["Synthetic data", "Dashboard", "Chatbot", "Equations & export"]
)

with tab_data:
    st.subheader("Synthetic 1 Hz time series")
    st.write(
        "This tab creates the same dataframe used by the dashboard and chatbot. The synthetic data contain "
        "placeholder role-level bio-behavioral features, symbolic role states, a symbolic team state, phase labels, "
        "and a synthetic event marker."
    )
    render_metric_cards(bundle)
    st.markdown("**Current synthetic time-series preview**")
    st.dataframe(bundle["timeseries_df"].head(100), use_container_width=True)
    with st.expander("Symbolic state table used for entropy, sample entropy, and AMI"):
        st.dataframe(bundle["states_df"].head(100), use_container_width=True)
    st.download_button(
        "Download current synthetic time series CSV",
        data=bundle["timeseries_df"].to_csv(index=False).encode("utf-8"),
        file_name="synthetic_biobehavioral_timeseries.csv",
        mime="text/csv",
    )

with tab_dashboard:
    st.subheader("Analysis dashboard for the current synthetic dataset")
    st.write(
        "The figures and tables below are computed from the synthetic data currently shown in the Synthetic data tab. "
        "The chatbot uses these same outputs when answering questions."
    )
    render_metric_cards(bundle)
    c1, c2 = st.columns(2)
    with c1:
        render_figures(["team_state", "entropy", "ami_summary"], bundle)
    with c2:
        render_figures(["inverse_sampen", "ami_heatmap"], bundle)

    st.subheader("Metric tables")
    table_choice = st.selectbox(
        "Choose a table",
        options=["entropy", "inverse_sampen", "ami_summary", "ami_long"],
        format_func=lambda key: TABLE_SPECS[key]["title"],
    )
    render_tables([table_choice], bundle, max_rows=150)

with tab_chat:
    st.subheader("Chatbot connected to the current synthetic data and analyses")
    st.write(
        "Ask the chatbot to run analyses, show figures, display tables, or explain a metric. Python computes the "
        "outputs locally; Gemini, when configured, explains the already-computed results."
    )
    st.markdown(
        "Try: `Run all analyses and show figures and tables`, `Explain the entropy peak`, "
        "`Show the AMI table`, or `Generate a new synthetic dataset with seed 7, 1200 seconds, event 600`."
    )

    for message in st.session_state.messages:
        render_message(message, bundle)

    user_question = st.chat_input("Ask the connected chatbot about the current synthetic data or analyses")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question, "artifacts": {}})
        request = classify_user_request(user_question)

        with st.spinner("Processing request with the connected Python analysis pipeline..."):
            if request["generate"]:
                updated_settings = parse_generation_overrides(user_question, settings)
                generate_data_and_run_analysis(updated_settings, user_question=user_question)
                bundle = st.session_state.analysis_bundle
                settings = app_settings_from_session()
            elif request["run_analysis"]:
                # Rerun analyses on the current connected synthetic dataframe.
                st.session_state.analysis_bundle = run_analysis_for_dataframe(
                    st.session_state.timeseries_df,
                    entropy_window=int(settings["entropy_window"]),
                    sampen_window=int(settings["sampen_window"]),
                    ami_window=int(settings["ami_window"]),
                    step=int(settings["step"]),
                    model=str(settings["model"]),
                    user_question=user_question,
                )
                bundle = st.session_state.analysis_bundle

        figure_keys = request["figure_keys"]
        table_keys = request["table_keys"]
        if request["run_analysis"] and not figure_keys and not table_keys:
            figure_keys = ["entropy", "inverse_sampen", "ami_summary"]
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
                response += "\n\nGemini is not configured, so this response is the local Python-computed summary only."
            elif not request["wants_gemini_explanation"]:
                response += "\n\nI displayed the requested computed artifacts without asking Gemini for a narrative explanation."

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "artifacts": {"figures": figure_keys, "tables": table_keys},
            }
        )
        st.rerun()

with tab_equations:
    st.subheader("Notebook equations and lightweight exports")
    st.write(
        "The Jupyter notebook remains credential-free and contains the equations for symbolic-state construction, "
        "Shannon entropy, sample entropy, inverse sample entropy, mutual information, AMI share, and HHI."
    )
    st.markdown(
        r"""
**Shannon entropy**

$$H_w = - \sum_{s \in \mathcal{S}} p_w(s) \log_2 p_w(s)$$

**Sample entropy**

$$\mathrm{SampEn}(m,N) = -\ln\left(\frac{A}{B}\right)$$

**Inverse sample entropy index**

$$\mathrm{InvSampEn} = \frac{1}{1 + \mathrm{SampEn}}$$

**Mutual information**

$$I(X;Y)=\sum_x\sum_y p(x,y)\log_2\frac{p(x,y)}{p(x)p(y)}$$

**Relative AMI share**

$$\mathrm{Share}_{r,w}=\frac{AMI_{r,w}}{\sum_j AMI_{j,w}}$$
        """
    )
    with st.expander("Download current outputs"):
        downloads = {
            "Synthetic time series CSV": bundle["timeseries_df"].to_csv(index=False),
            "Entropy table CSV": bundle["entropy_df"].to_csv(index=False),
            "Inverse sample entropy table CSV": bundle["inverse_df"].to_csv(index=False),
            "AMI summary table CSV": bundle["ami_summary_df"].to_csv(index=False),
            "Explanation packet JSON": json.dumps(bundle["packet"], indent=2),
        }
        for label, content in downloads.items():
            suffix = "json" if label.endswith("JSON") else "csv"
            st.download_button(
                label,
                data=content.encode("utf-8"),
                file_name=label.lower().replace(" ", "_").replace("/", "_") + f".{suffix}",
                mime="application/json" if suffix == "json" else "text/csv",
            )
    with st.expander("Current Gemini prompt payload"):
        prompt_path = bundle["paths"].prompt_md
        if prompt_path.exists():
            st.code(prompt_path.read_text(encoding="utf-8")[:12000], language="markdown")
        else:
            st.info("Ask the chatbot a question to create the current Gemini prompt payload.")
