"""Streamlit app for the Bio-Behavioral Team Dynamics Analytics Chatbot prototype.

Run from the repository root:
    streamlit run streamlit_app.py

API-key handling:
    - Local development: copy .streamlit/secrets.toml.example to .streamlit/secrets.toml.
    - Streamlit Community Cloud: add the key in the app's Secrets settings.
    - The notebook does not request, display, or store any API key.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from team_dynamics_metrics import (  # noqa: E402
    DEFAULT_USER_QUESTION,
    build_gemini_prompt,
    call_gemini,
    load_identity_text,
    run_pipeline,
)

IDENTITY_PATH = ROOT / "docs" / "team_dynamics_chatbot_identity.md"
OUTPUT_DIR = ROOT / "outputs"


def get_gemini_key() -> Optional[str]:
    """Read Gemini key from Streamlit secrets, then fall back to environment variables."""

    secret_key = None
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        secret_key = None
    return secret_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def ensure_demo_outputs(
    seed: int = 42,
    n_seconds: int = 900,
    event_time_sec: int = 450,
    entropy_window: int = 120,
    sampen_window: int = 180,
    ami_window: int = 180,
    step: int = 15,
    model: str = "gemini-2.5-flash",
) -> None:
    """Generate synthetic data and metric outputs if they are missing."""

    packet_path = OUTPUT_DIR / "team_dynamics_explanation_packet.json"
    core_files = [
        packet_path,
        OUTPUT_DIR / "synthetic_biobehavioral_timeseries.csv",
        OUTPUT_DIR / "tables" / "moving_window_entropy.csv",
        OUTPUT_DIR / "tables" / "moving_window_inverse_sample_entropy.csv",
        OUTPUT_DIR / "tables" / "role_ami_summary.csv",
        OUTPUT_DIR / "figures" / "synthetic_entropy_trajectory.png",
        OUTPUT_DIR / "figures" / "synthetic_inverse_sampen_trajectory.png",
        OUTPUT_DIR / "figures" / "synthetic_role_ami_summary.png",
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in core_files):
        run_pipeline(
            output_dir=OUTPUT_DIR,
            identity_path=IDENTITY_PATH,
            seed=seed,
            n_seconds=n_seconds,
            event_time_sec=event_time_sec,
            entropy_window=entropy_window,
            sampen_window=sampen_window,
            ami_window=ami_window,
            step=step,
            model=model,
        )


def load_packet() -> dict:
    ensure_demo_outputs()
    packet_path = OUTPUT_DIR / "team_dynamics_explanation_packet.json"
    return json.loads(packet_path.read_text(encoding="utf-8"))


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


st.set_page_config(
    page_title="Team Dynamics Analytics Chatbot",
    page_icon="💬",
    layout="wide",
)

st.title("Bio-Behavioral Team Dynamics Analytics Chatbot Prototype")
st.caption(
    "Synthetic demonstration data only. The prototype supports the dissertation RQ2 goal of helping users "
    "understand metric derivations, visual outputs, and limits of interpretation. It does not display BioTDMS "
    "data or dissertation findings."
)

with st.sidebar:
    st.header("Synthetic time-series settings")
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
    n_seconds = st.slider("Synthetic duration (seconds, 1 Hz)", min_value=300, max_value=1800, value=900, step=60)
    event_time_sec = st.slider(
        "Synthetic event time (seconds)",
        min_value=30,
        max_value=n_seconds - 30,
        value=min(450, n_seconds - 30),
        step=15,
    )
    entropy_window = st.slider("Entropy window (seconds)", min_value=60, max_value=300, value=120, step=15)
    sampen_window = st.slider("Sample entropy window (seconds)", min_value=90, max_value=360, value=180, step=15)
    ami_window = st.slider("AMI window (seconds)", min_value=90, max_value=360, value=180, step=15)
    step = st.slider("Window step (seconds)", min_value=5, max_value=60, value=15, step=5)
    model = st.text_input("Gemini model", value="gemini-2.5-flash")

    if st.button("Generate synthetic data and run analyses", type="primary"):
        with st.spinner("Generating synthetic time series, metrics, figures, and Gemini prompt payload..."):
            run_pipeline(
                output_dir=OUTPUT_DIR,
                identity_path=IDENTITY_PATH,
                seed=int(seed),
                n_seconds=int(n_seconds),
                event_time_sec=int(event_time_sec),
                entropy_window=int(entropy_window),
                sampen_window=int(sampen_window),
                ami_window=int(ami_window),
                step=int(step),
                model=model,
            )
        st.session_state.pop("packet", None)
        st.success("Synthetic data, metric tables, figures, and prompt payload regenerated.")

    st.divider()
    st.header("API status")
    api_key = get_gemini_key()
    if api_key:
        st.success("Gemini key detected through Streamlit secrets or environment variables.")
    else:
        st.warning("No Gemini key detected. Analyses and figures still work; chatbot responses require a key.")
        st.caption("Do not paste the key into a notebook. Use `.streamlit/secrets.toml` or Streamlit Cloud secrets.")

if "packet" not in st.session_state:
    st.session_state.packet = load_packet()
packet = st.session_state.packet

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Ask me about the synthetic time series, entropy trajectory, inverse sample entropy trajectory, "
                "AMI influence profile, or what these metrics can and cannot support."
            ),
        }
    ]

tab_dashboard, tab_data, tab_chat, tab_equations, tab_files = st.tabs(
    ["Dashboard", "Synthetic data", "Chatbot", "Equations", "Generated files"]
)

with tab_dashboard:
    st.subheader("Synthetic analysis outputs")
    st.write(
        "These figures are generated from a synthetic 1 Hz time series. They are used to test the public "
        "analysis and explanation workflow, not to report dissertation findings."
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(OUTPUT_DIR / "figures" / "synthetic_team_state_trajectory.png"), caption="Symbolic team-state trajectory")
        st.image(str(OUTPUT_DIR / "figures" / "synthetic_entropy_trajectory.png"), caption="Moving-window Shannon entropy")
        st.image(str(OUTPUT_DIR / "figures" / "synthetic_role_ami_summary.png"), caption="Role-level AMI influence summary")
    with col2:
        st.image(str(OUTPUT_DIR / "figures" / "synthetic_inverse_sampen_trajectory.png"), caption="Moving-window inverse sample entropy")
        st.image(str(OUTPUT_DIR / "figures" / "synthetic_role_influence_heatmap.png"), caption="Moving-window relative AMI influence")

    st.subheader("Metric table preview")
    table_name = st.selectbox(
        "Select a generated table",
        [
            "moving_window_entropy.csv",
            "moving_window_inverse_sample_entropy.csv",
            "role_ami_summary.csv",
            "moving_window_role_ami_long.csv",
        ],
    )
    table_path = OUTPUT_DIR / "tables" / table_name
    table_df = read_csv_if_exists(table_path)
    if not table_df.empty:
        st.dataframe(table_df, use_container_width=True)
    else:
        st.info("Table not found. Use the sidebar to generate outputs.")

with tab_data:
    st.subheader("Synthetic 1 Hz time series")
    st.write(
        "The public prototype creates synthetic role-level continuous features and symbolic state labels. "
        "The continuous features are placeholders that resemble analysis-ready summaries; they are not real physiological data."
    )
    data_path = OUTPUT_DIR / "synthetic_biobehavioral_timeseries.csv"
    state_path = OUTPUT_DIR / "synthetic_symbolic_states.csv"
    data_df = read_csv_if_exists(data_path)
    state_df = read_csv_if_exists(state_path)
    if not data_df.empty:
        st.metric("Rows", len(data_df))
        st.metric("Sampling rate", "1 Hz")
        st.dataframe(data_df.head(50), use_container_width=True)
        st.download_button(
            "Download synthetic time series CSV",
            data=data_df.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_biobehavioral_timeseries.csv",
            mime="text/csv",
        )
    if not state_df.empty:
        with st.expander("View symbolic state table used for entropy, sample entropy, and AMI"):
            st.dataframe(state_df.head(50), use_container_width=True)

with tab_chat:
    st.subheader("Chat with the synthetic-output explanation bot")
    st.write(
        "The chatbot receives the identity file, the synthetic JSON explanation packet, and your question. "
        "The Gemini API call happens only inside Streamlit after a key has been configured in secrets or the environment."
    )

    if not get_gemini_key():
        st.warning(
            "Gemini is not configured. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` locally or to Streamlit Cloud secrets. "
            "The app will not create a fake offline response."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask about the synthetic team dynamics outputs")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        identity_text = load_identity_text(IDENTITY_PATH)
        prompt = build_gemini_prompt(identity_text, packet, user_question=user_question)
        (OUTPUT_DIR / "gemini_prompt_payload.md").write_text(prompt, encoding="utf-8")

        if not get_gemini_key():
            response = (
                "Gemini is not configured yet. Add the API key through Streamlit secrets, then ask again. "
                "The synthetic analyses and prompt payload have been generated, but no offline substitute response was created."
            )
        else:
            with st.spinner("Calling Gemini..."):
                try:
                    response = call_gemini(prompt, model=model, api_key=get_gemini_key())
                    (OUTPUT_DIR / "gemini_chatbot_response.md").write_text(response, encoding="utf-8")
                except Exception as exc:  # UI should show a clean error, not expose secrets.
                    response = f"Gemini call failed: {exc}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    with st.expander("View approved JSON explanation packet"):
        st.json(packet)

with tab_equations:
    st.subheader("Equations included in the Jupyter notebook")
    st.markdown(
        r"""
The notebook contains the technical equations and runs the same synthetic pipeline without any API-key cell.

**Symbolic state construction**

$$S_t = f(R_{1t}, R_{2t}, \ldots, R_{kt})$$

where $R_{it}$ is a role-level symbolic state and $S_t$ is the synthetic team-state label.

**Shannon entropy**

$$H_w = - \sum_{s \in \mathcal{S}} p_w(s) \log_2 p_w(s)$$

**Normalized entropy**

$$H_w^* = \frac{H_w}{\log_2 K}$$

**Categorical sample entropy**

$$\mathrm{SampEn}(m,N) = -\ln\left(\frac{A}{B}\right)$$

**Inverse sample entropy index**

$$\mathrm{InvSampEn} = \frac{1}{1 + \mathrm{SampEn}}$$

**Mutual information**

$$I(X;Y)=\sum_x\sum_y p(x,y)\log_2\frac{p(x,y)}{p(x)p(y)}$$

**Lag-averaged AMI**

$$AMI_r = \frac{1}{L+1}\sum_{\ell=0}^{L} NMI(R_{r,t}, S_{t+\ell})$$

**Relative AMI influence share**

$$\mathrm{Share}_{r,w}=\frac{AMI_{r,w}}{\sum_j AMI_{j,w}}$$

**Influence concentration**

$$HHI_w=\sum_r \mathrm{Share}_{r,w}^2$$
        """
    )
    st.info("Open `notebooks/team_dynamics_chatbot_equations.ipynb` for the full technical component.")

with tab_files:
    st.subheader("Generated public artifacts")
    files = [
        OUTPUT_DIR / "synthetic_biobehavioral_timeseries.csv",
        OUTPUT_DIR / "synthetic_symbolic_states.csv",
        OUTPUT_DIR / "team_dynamics_explanation_packet.json",
        OUTPUT_DIR / "gemini_prompt_payload.md",
        OUTPUT_DIR / "api_status.json",
        OUTPUT_DIR / "tables" / "moving_window_entropy.csv",
        OUTPUT_DIR / "tables" / "moving_window_inverse_sample_entropy.csv",
        OUTPUT_DIR / "tables" / "role_ami_summary.csv",
        OUTPUT_DIR / "tables" / "moving_window_role_ami_long.csv",
        OUTPUT_DIR / "figures" / "synthetic_team_state_trajectory.png",
        OUTPUT_DIR / "figures" / "synthetic_entropy_trajectory.png",
        OUTPUT_DIR / "figures" / "synthetic_inverse_sampen_trajectory.png",
        OUTPUT_DIR / "figures" / "synthetic_role_ami_summary.png",
        OUTPUT_DIR / "figures" / "synthetic_role_influence_heatmap.png",
    ]
    for path in files:
        status = "available" if path.exists() and path.stat().st_size > 0 else "missing"
        st.write(f"- `{path.relative_to(ROOT)}` - {status}")

    with st.expander("Current Gemini prompt payload"):
        prompt_path = OUTPUT_DIR / "gemini_prompt_payload.md"
        if prompt_path.exists():
            st.code(prompt_path.read_text(encoding="utf-8")[:12000], language="markdown")
        else:
            st.info("Prompt payload not generated yet.")
