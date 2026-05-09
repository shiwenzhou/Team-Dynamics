# Bio-Behavioral Team Dynamics Analytics Chatbot Prototype

This repository is a public-facing course prototype for a dissertation RQ2 chatbot. RQ2 asks whether a customized AI bot can help non-experts understand and interpret bio-behavioral team dynamics analytics, including how metrics are derived and how they lead to reported results.

The project uses **synthetic demonstration data only**. It does not include BioTDMS data, raw physiological records, audio, transcripts, identifiable trainee information, or dissertation findings.

## What works immediately

After installing the requirements, the Streamlit app can immediately:

1. Generate a synthetic 1 Hz multimodal team time series.
2. Compute moving-window Shannon entropy for adaptation/reorganization.
3. Compute categorical sample entropy and inverse sample entropy for interdependence-related temporal structure.
4. Compute lag-averaged role-level AMI, AMI share, and HHI concentration for influence distribution.
5. Produce metric tables, figures, and a JSON explanation packet.
6. Build a Gemini prompt payload for the chatbot.

The **chatbot response** requires a Gemini API key configured through Streamlit secrets or environment variables. The app does not create a fake offline LLM answer.

## Repository structure

```text
.
├── streamlit_app.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── team_dynamics_metrics.py
├── notebooks/
│   └── team_dynamics_chatbot_equations.ipynb
├── docs/
│   ├── team_dynamics_chatbot_identity.md
│   ├── team_dynamics_prompt_log.md
│   └── Zhou_S_Final_Paper_Team_Dynamics_Streamlit.docx
├── outputs/
│   ├── synthetic_biobehavioral_timeseries.csv
│   ├── synthetic_symbolic_states.csv
│   ├── team_dynamics_explanation_packet.json
│   ├── gemini_prompt_payload.md
│   ├── api_status.json
│   ├── figures/
│   └── tables/
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Gemini API key setup

Do **not** paste the API key into the Jupyter notebook. Use one of these options.

### Local Streamlit secrets

Copy the example file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

The real `.streamlit/secrets.toml` file is ignored by Git.

### Environment variable

```bash
export GEMINI_API_KEY="your_api_key_here"
streamlit run streamlit_app.py
```

## Jupyter notebook

The notebook is the technical component for the course. It includes equations for:

- symbolic state construction
- Shannon entropy
- normalized entropy
- categorical sample entropy
- inverse sample entropy
- mutual information
- normalized mutual information
- lag-averaged AMI
- AMI influence share
- HHI influence concentration

It also runs the synthetic pipeline and displays the generated outputs. It does not request or store an API key.

## Streamlit app workflow

The app has five tabs:

1. **Dashboard** - shows generated figures and metric tables.
2. **Synthetic data** - previews and downloads the synthetic 1 Hz time series.
3. **Chatbot** - sends the identity file, JSON explanation packet, and user question to Gemini when an API key is configured.
4. **Equations** - summarizes the equations and points to the notebook.
5. **Generated files** - lists generated output artifacts.

## Important interpretation limits

The synthetic data are for debugging, explanation design, and GitHub display. They are not evidence about real team performance. The metrics are descriptive. Entropy, inverse sample entropy, and AMI should not be interpreted as direct proof of better or worse teamwork, individual competence, stress, workload, or causal influence.
