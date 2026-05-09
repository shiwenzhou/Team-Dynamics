# Bio-Behavioral Team Dynamics Analytics Chatbot Prototype

This repository is a public, synthetic-data Streamlit prototype for a dissertation-support chatbot. The dissertation aim supported here is the customized AI bot portion: helping non-experts understand bio-behavioral team dynamics analytics, including interdependence, adaptation, influence distribution, metric derivations, visual outputs, and limits of interpretation.

The app does **not** use BioTDMS data and does **not** report dissertation findings. All data generated in this repository are synthetic demonstration data.

## What the app does

The Streamlit app now uses one connected workflow:

1. **Synthetic data** generates a synthetic 1 Hz time series for a five-role Fire Support Team-style demonstration.
2. **Dashboard** computes and displays the analysis results from that same synthetic dataset.
3. **Chatbot** can trigger the local Python analyses and display the current figures and tables in the chat response. When a Gemini key is configured, Gemini explains the already-computed outputs.

The chatbot does not independently calculate metrics. Python computes the metrics; Gemini receives a structured explanation packet and helps translate the outputs into plain language.

## Analyses included

The prototype computes:

- Moving-window Shannon entropy for adaptation / reorganization.
- Moving-window categorical sample entropy and inverse sample entropy for temporal regularity / interdependence demonstration.
- Lag-averaged normalized mutual information (AMI) between role symbolic states and the team-state sequence for influence distribution.
- AMI share and HHI concentration as descriptive summaries of influence distribution.

## Repository structure

```text
.
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── docs/
│   ├── team_dynamics_chatbot_identity.md
│   └── team_dynamics_prompt_log.md
├── notebooks/
│   └── team_dynamics_chatbot_equations.ipynb
├── src/
│   ├── __init__.py
│   └── team_dynamics_metrics.py
└── outputs/
    ├── synthetic_biobehavioral_timeseries.csv
    ├── synthetic_symbolic_states.csv
    ├── team_dynamics_explanation_packet.json
    ├── tables/
    └── figures/
```

## Local setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

The analysis pipeline can also be run from the command line:

```bash
python src/team_dynamics_metrics.py --output-dir outputs --identity-path docs/team_dynamics_chatbot_identity.md
```

## Gemini API key setup

The notebook does not contain or request an API key. The Streamlit app reads the key from Streamlit secrets or environment variables.

For local development, copy the example secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then edit `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

Do not commit `.streamlit/secrets.toml`. The `.gitignore` file excludes it.

You can also set an environment variable:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Example chatbot questions

Try these in the Chatbot tab:

- `Run all analyses and show figures and tables.`
- `Explain the entropy peak.`
- `Show the AMI table.`
- `What does inverse sample entropy mean here?`
- `Generate a new synthetic dataset with seed 7, 1200 seconds, event 600.`

## Important interpretation limits

The public version uses synthetic data. The outputs are suitable for testing the analysis and explanation workflow, not for drawing conclusions about real teams or trainees. The chatbot identity file instructs the model not to infer mental states, not to rank individual trainees, not to treat entropy or AMI as direct evidence of good or bad teamwork, and not to fabricate unsupported results.
