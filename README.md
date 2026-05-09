# Bio-Behavioral Team Dynamics Analytics Chatbot Prototype

This repository contains a Streamlit prototype for a course final project connected to the dissertation aim of building a customized AI bot that helps stakeholders understand bio-behavioral team dynamics analytics. The public version uses **synthetic data only**. It does not use BioTDMS data, does not test dissertation hypotheses, and does not evaluate real trainees.

## App layout

The tab order is back to the original order:

1. **Dashboard**
2. **Synthetic data**
3. **Chatbot**

The three tabs are connected through one `st.session_state.analysis_bundle`. The current synthetic dataframe, metric tables, figures, JSON packet, dataset ID, and settings are stored together. When the synthetic data are changed, the Dashboard and Chatbot use the new dataset ID.

## How the connection works

1. Open the **Synthetic data** tab.
2. Change the seed, duration, event time, or analysis-window settings.
3. Click **Generate synthetic data + run analyses**.
4. Return to **Dashboard** or **Chatbot**. The same dataset ID should appear in all tabs.
5. Ask the Chatbot to run analyses or show figures/tables. It uses the current synthetic dataframe from the Synthetic data tab.

Example chatbot prompt:

```text
Run all analyses and show figures and tables using the current synthetic data.
```

Another example:

```text
Generate a new synthetic dataset with seed 7, 1200 seconds, event 600, then run all analyses.
```

## Analyses included

The local Python pipeline computes the following from the synthetic 1 Hz time series:

- Symbolic team-state trajectory
- Moving-window Shannon entropy
- Moving-window sample entropy and inverse sample entropy
- Moving-window role-level average mutual information (AMI)
- Role-level AMI summary and influence-concentration HHI
- Figures and tables displayed directly in the Dashboard and Chatbot
- A Gemini prompt payload for explanation of the computed outputs

Python performs the analyses. Gemini, when configured, explains the already-computed outputs; it does not run the calculations.

## Running locally

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

## Gemini key handling

Do not paste an API key into the notebook. The notebook is credential-free.

For local Streamlit use, create `.streamlit/secrets.toml` and add:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

For Streamlit Community Cloud, add the same key in the app's secrets settings. The app also checks the environment variables `GEMINI_API_KEY` and `GOOGLE_API_KEY`.

## Repository structure

```text
.
├── streamlit_app.py
├── requirements.txt
├── src/
│   └── team_dynamics_metrics.py
├── docs/
│   ├── team_dynamics_chatbot_identity.md
│   └── team_dynamics_prompt_log.md
├── notebooks/
│   └── team_dynamics_chatbot_equations.ipynb
├── outputs/
│   ├── synthetic_biobehavioral_timeseries.csv
│   ├── team_dynamics_explanation_packet.json
│   ├── figures/
│   └── tables/
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Interpretation boundaries

The figures and tables are synthetic demonstration artifacts. They are intended to test whether the UI, metric pipeline, prompt structure, and chatbot explanation workflow connect correctly. They should not be interpreted as evidence about real Fire Support Teams, real trainees, or actual performance.
