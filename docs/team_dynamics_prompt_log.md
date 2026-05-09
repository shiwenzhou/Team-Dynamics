# Self-Generated Prompt Log for the Connected Bio-Behavioral Team Dynamics Chatbot Prototype

This prompt log documents the AI-assisted development workflow for the final Streamlit prototype. These entries are written as standalone prompts that could be submitted to ChatGPT or Claude during code generation, debugging, interface design, interpretation design, and documentation. They are not copied from an informal chat window. Each entry includes a validation plan to make the workflow transparent and human-checked.

## Prompt 1 - ChatGPT - Reframe the prototype around the updated dissertation proposal

**Purpose:** Align the project with the dissertation language of bio-behavioral team dynamics analytics, rather than treating the app as a general teamness dashboard.

**Prompt:**
You are helping me revise a course final project for a prompt-engineering and data-analysis class. My dissertation proposal frames the work as objective bio-behavioral measurement of team interdependence, adaptation, and influence distribution, plus a customized AI bot for interpretation. The course artifact should support the AI-bot portion of the dissertation. Rewrite the prototype scope so it presents itself as a bio-behavioral team dynamics analytics tool. Make the goal clear: a chatbot that helps non-experts understand dynamic team metrics, how they are derived, what the figures show, and what cannot be concluded. Use academic but accessible wording.

**Validation:** I checked that the revised wording used bio-behavioral team dynamics analytics and the three constructs of interdependence, adaptation, and influence distribution. I removed interface language that made the project sound like it was testing dissertation hypotheses or reporting dissertation results.

## Prompt 2 - Claude - Review identity-file safeguards

**Purpose:** Improve the chatbot identity file for a stakeholder-facing interpretation assistant.

**Prompt:**
Act as a careful human-factors and explainable-AI reviewer. Review this identity file for a chatbot that explains synthetic or de-identified bio-behavioral team dynamics analytics. Identify missing safeguards around privacy, unsupported inference, physiological data interpretation, individual ranking, causal claims, fabricated citations, and synthetic demonstration data. Suggest edits that keep the bot conversational while preventing over-interpretation. Do not make the bot sound like a military evaluator or a validated performance-scoring system.

**Validation:** I kept safeguards that aligned with the dissertation RQ2 interpretation role: no raw data requests, no mental-state inference, no individual evaluation, no causal claims, no fabricated results, and explicit warnings that synthetic data are not dissertation findings.

## Prompt 3 - ChatGPT - Generate a synthetic 1 Hz team dynamics dataset

**Purpose:** Create public demonstration data that can be shown on GitHub without exposing BioTDMS data.

**Prompt:**
Write Python functions using pandas and numpy to generate a synthetic 1 Hz time series for a five-role Fire Support Team-style demonstration. Each row should represent one second. For each role, create analysis-ready placeholder features for heart rate, respiration rate, EEG alpha power, gaze-task focus, communication activity, and a symbolic role state. Also create a symbolic team state, phase label, and a synthetic event marker. The data must be clearly labeled synthetic and must not look like real participant data. Use deterministic seeding and write validation checks for required columns.

**Validation:** I ran the data generator from a clean shell, confirmed that `synthetic_biobehavioral_timeseries.csv` and `synthetic_symbolic_states.csv` were created, and checked that the files include only synthetic identifiers and no real BioTDMS values.

## Prompt 4 - ChatGPT - Implement the analytic functions

**Purpose:** Add the required computational component for the course prototype.

**Prompt:**
Write well-organized Python functions for a synthetic team dynamics prototype. Include moving-window Shannon entropy for adaptation, categorical sample entropy and inverse sample entropy for interdependence, lag-averaged normalized mutual information for influence distribution, AMI influence share, and HHI concentration. Use only pandas, numpy, and matplotlib. Save output tables and figures. Add validation checks that entropy values are nonnegative, normalized entropy is in [0, 1], AMI shares sum to 1 within each window, and all expected files are created.

**Validation:** I ran the pipeline with default settings and confirmed that it produced the synthetic time-series CSV, metric tables, PNG figures, JSON explanation packet, Gemini prompt payload, and API status file. I also checked that AMI shares sum to 1 within each moving window.

## Prompt 5 - Claude - Diagnose why the first Streamlit layout felt disconnected

**Purpose:** Fix the app so Synthetic data, Dashboard, and Chatbot are connected rather than acting like separate pages.

**Prompt:**
Audit this Streamlit prototype. It currently has Synthetic data, Dashboard, Chatbot, and Generated files tabs, but the chatbot does not feel connected to the synthetic data the user generated. Redesign the app around one shared session state. The Synthetic data tab should generate the dataframe, the Dashboard should compute and display analyses from that same dataframe, and the Chatbot should be able to trigger analyses and display figures/tables in the chat response. Remove the separate Generated files tab because it does not help the user understand the workflow. Keep downloads available only as a lightweight export section.

**Validation:** I rewrote the app so `st.session_state.timeseries_df` and `st.session_state.analysis_bundle` are the single source of truth. Regenerating synthetic data reruns all analyses. The chatbot can now show the same entropy, inverse sample entropy, and AMI figures/tables that appear on the dashboard.

## Prompt 6 - ChatGPT - Build a local analysis controller for the chatbot

**Purpose:** Allow the chatbot to run analyses and display outputs, while keeping metric computation in Python.

**Prompt:**
Write a Streamlit chat controller that classifies user requests such as "run all analyses," "show entropy," "display the AMI table," and "generate a new synthetic dataset with seed 7, 1200 seconds, event 600." The controller should call local Python analysis functions, update the current analysis bundle, and attach relevant figures and tables to the assistant message. Gemini should only explain already-computed outputs. It should not be asked to calculate entropy or produce figures.

**Validation:** I added request classification for generate, run analysis, entropy, inverse sample entropy, AMI, figures, and tables. I checked that assistant messages can include rendered Streamlit images and dataframes from the current analysis bundle.

## Prompt 7 - Claude - Audit API-key handling for GitHub

**Purpose:** Make the GitHub version safe to share without exposing a key.

**Prompt:**
Review the Gemini API-key handling in this Streamlit app. The app should read the key from `st.secrets` or from environment variables, should never hard-code a key, should not request the key in the Jupyter notebook, and should not commit `.streamlit/secrets.toml` to GitHub. Suggest concise changes and README language for local and Streamlit Cloud setup.

**Validation:** I confirmed that the repository includes `.streamlit/secrets.toml.example` but not a real secrets file, that `.gitignore` excludes `.streamlit/secrets.toml`, and that the code reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` without printing or saving the secret value.

## Prompt 8 - ChatGPT - Build the Gemini explanation prompt for connected outputs

**Purpose:** Ensure Gemini explains the exact outputs computed by the app.

**Prompt:**
Using the identity file and this JSON explanation packet, create a reusable Gemini prompt payload for a chatbot that explains bio-behavioral team dynamics analytics. The prompt should make clear that local Python functions already generated the synthetic data, metric tables, and figures. Gemini's job is to answer the user, explain relevant concepts in plain language, connect the answer to the supplied metric summaries and figure captions, and state what cannot be concluded. Do not include raw data. Do not infer mental states. Do not rank individual trainees. Do not invent calculations.

**Validation:** I inspected the generated prompt payload and confirmed that it includes the identity text, user question, JSON packet, output structure, synthetic-data warning, and constraints against individual evaluation and causal overclaiming.

## Prompt 9 - ChatGPT - Create equations for the notebook

**Purpose:** Ensure the technical component includes formulas, not just code.

**Prompt:**
Write notebook-ready Markdown equations for the synthetic bio-behavioral team dynamics prototype. Include symbolic state construction, Shannon entropy, normalized entropy, categorical sample entropy, inverse sample entropy, mutual information, normalized mutual information, lag-averaged AMI, AMI influence share, and HHI concentration. After each equation, add a short plain-language explanation of how it connects to the prototype.

**Validation:** I inserted the equations into the Jupyter notebook and checked that the notebook runs the synthetic pipeline without requiring an API key.

## Prompt 10 - Claude - Final GitHub readiness check

**Purpose:** Make sure the project is understandable and reproducible for course submission.

**Prompt:**
Act as a GitHub repository reviewer. Review the folder structure for a Streamlit prototype that uses synthetic data and a Jupyter notebook. Check whether the README explains installation, how to run the app, how the synthetic data connect to the dashboard and chatbot, how to configure Gemini secrets, what files should not be committed, and how the prototype connects to the dissertation RQ2 chatbot goal. Suggest any missing files or unclear instructions.

**Validation:** I confirmed that the package includes `streamlit_app.py`, `src/team_dynamics_metrics.py`, an executed notebook, an identity file, a prompt log, generated synthetic outputs, `requirements.txt`, `.gitignore`, `.streamlit/secrets.toml.example`, and a README. I also removed the separate Generated files tab and replaced it with a small export section.
