# Self-Generated Prompt Log for the Bio-Behavioral Team Dynamics Chatbot Prototype

This prompt log documents the AI-assisted development workflow for the course prototype. The entries are written as standalone prompts that could be submitted to ChatGPT or Claude during code generation, debugging, interface design, interpretation design, and documentation. They are not copied from an informal chat window. Each entry includes a validation plan to make the workflow transparent and human-checked.

## Prompt 1 - ChatGPT - Reframe the prototype around the updated dissertation proposal

**Purpose:** Replace the earlier prototype wording with the updated dissertation language: bio-behavioral team dynamics analytics focused on interdependence, adaptation, and influence distribution.

**Prompt:**
You are helping me revise a course final project for a prompt-engineering and data-analysis class. My updated dissertation proposal frames the work as objective bio-behavioral measurement of team interdependence, adaptation, and influence distribution, plus a customized AI bot for interpretation. The course artifact should support the AI-bot portion of the dissertation. Rewrite the prototype scope so it presents itself as a bio-behavioral team dynamics analytics tool rather than using older wording. Make the goal clear: a chatbot that helps non-experts understand dynamic team metrics, how they are derived, what the figures show, and what cannot be concluded. Use academic but accessible wording.

**Validation:** I checked that the revised scope used "bio-behavioral team dynamics analytics" and the three constructs of interdependence, adaptation, and influence distribution. I removed the prototype's earlier title and variable descriptions that did not match the updated dissertation wording.

## Prompt 2 - Claude - Review identity file safeguards

**Purpose:** Improve the chatbot identity file for a stakeholder-facing interpretation assistant.

**Prompt:**
Act as a careful human-factors and explainable-AI reviewer. Review this identity file for a chatbot that explains synthetic or de-identified bio-behavioral team dynamics analytics. Identify missing safeguards around privacy, unsupported inference, physiological data interpretation, individual ranking, causal claims, fabricated citations, and synthetic demonstration data. Suggest edits that keep the bot conversational while preventing over-interpretation. Do not make the bot sound like a military evaluator or a validated performance-scoring system.

**Validation:** I kept safeguards that aligned with the dissertation RQ2 interpretation role: no raw data requests, no mental-state inference, no individual evaluation, no causal claims, no fabricated results, and explicit warnings that synthetic data are not dissertation findings.

## Prompt 3 - ChatGPT - Generate a synthetic 1 Hz multimodal team time series

**Purpose:** Create public demonstration data that can be shown on GitHub without exposing BioTDMS data.

**Prompt:**
Write Python functions using pandas and numpy to generate a synthetic 1 Hz time series for a five-role Fire Support Team demonstration. Each row should represent one second. For each role, create analysis-ready placeholder features for heart rate, respiration rate, EEG alpha power, gaze-task focus, communication activity, and a symbolic role state. Also create a symbolic team state, phase label, and a synthetic event marker. The data must be clearly labeled synthetic and must not look like real participant data. Use deterministic seeding and write validation checks for required columns.

**Validation:** I ran the data generator from a clean shell, confirmed that `synthetic_biobehavioral_timeseries.csv` and `synthetic_symbolic_states.csv` were created, and checked that the files include only synthetic identifiers and no real BioTDMS values.

## Prompt 4 - ChatGPT - Implement the analytic functions

**Purpose:** Add the required computational component for the course prototype.

**Prompt:**
Write well-organized Python functions for a synthetic team dynamics prototype. Include moving-window Shannon entropy for adaptation, categorical sample entropy and inverse sample entropy for interdependence, lag-averaged normalized mutual information for influence distribution, AMI influence share, and HHI concentration. Use only pandas, numpy, and matplotlib. Save output tables and figures. Add validation checks that entropy values are nonnegative, normalized entropy is in [0, 1], AMI shares sum to 1 within each window, and all expected files are created.

**Validation:** I ran the pipeline with default settings and confirmed that it produced the synthetic time-series CSV, metric tables, PNG figures, JSON explanation packet, Gemini prompt payload, and API status file. I also checked that AMI shares sum to 1 within each moving window.

## Prompt 5 - Claude - Debug Streamlit app organization

**Purpose:** Move the interactive prototype out of the notebook so no API key is requested or stored in the notebook.

**Prompt:**
Audit this Streamlit prototype for a course project. The app should generate synthetic data, run the analyses, display figures and tables, and send a chatbot prompt to Gemini only after a key is configured through Streamlit secrets or environment variables. The Jupyter notebook should contain equations and synthetic analysis only. Identify hidden state assumptions, import problems, path problems, missing output checks, and places where the app might accidentally create a fake offline LLM response.

**Validation:** I separated the API call into the Streamlit app, kept the notebook credential-free, used `.streamlit/secrets.toml.example` for local setup, and ensured that the missing-key behavior is an explicit message rather than a mock chatbot answer.

## Prompt 6 - ChatGPT - Build a Gemini prompt payload for conversational explanation

**Purpose:** Construct the structured prompt used by the chatbot tab.

**Prompt:**
Using the identity file and this JSON explanation packet, create a reusable Gemini prompt payload for a chatbot that explains bio-behavioral team dynamics analytics. The chatbot should answer a user's question, explain relevant concepts in plain language, connect the answer to the supplied metric summaries and figure captions, and state what cannot be concluded. The prompt must include role, context, constraints, data packet, output structure, uncertainty rules, and privacy safeguards. Do not include raw data. Do not infer mental states. Do not rank individual trainees. Do not generate an offline substitute response.

**Validation:** I inspected the generated prompt payload and confirmed that it includes the identity text, user question, JSON packet, output structure, synthetic-data warning, and constraints against individual evaluation and causal overclaiming.

## Prompt 7 - Claude - Audit API-key handling for GitHub

**Purpose:** Make the GitHub version safe to share without exposing a key.

**Prompt:**
Review the Gemini API-key handling in this Streamlit app. The app should read the key from `st.secrets` or from environment variables, should never hard-code a key, should not request the key in the Jupyter notebook, and should not commit `.streamlit/secrets.toml` to GitHub. Suggest concise changes and README language for local and Streamlit Cloud setup.

**Validation:** I confirmed that the repository includes `.streamlit/secrets.toml.example` but not a real secrets file, that `.gitignore` excludes `.streamlit/secrets.toml`, and that the code reads `GEMINI_API_KEY` or `GOOGLE_API_KEY` without printing or saving the secret value.

## Prompt 8 - ChatGPT - Create equations for the notebook

**Purpose:** Ensure the technical component includes formulas, not just code.

**Prompt:**
Write notebook-ready Markdown equations for the synthetic team dynamics prototype. Include symbolic state construction, Shannon entropy, normalized entropy, categorical sample entropy, inverse sample entropy, mutual information, normalized mutual information, lag-averaged AMI, AMI influence share, and HHI concentration. After each equation, add a short plain-language explanation of how it connects to the prototype.

**Validation:** I inserted the equations into the Jupyter notebook and checked that the notebook runs the same synthetic pipeline without requiring an API key.

## Prompt 9 - ChatGPT - Generate stakeholder-style test questions

**Purpose:** Prepare questions for future API testing in the Streamlit chat tab.

**Prompt:**
Generate five stakeholder-style questions that could be asked in a chatbot interface for a bio-behavioral team dynamics dashboard. The questions should cover concept definitions, figure interpretation, metric derivation, limitations, and human validation checks. Do not ask for real participant data.

**Validation:** I used the questions to check whether the identity file and prompt payload support conversational user interaction while staying within privacy and interpretation limits.

## Prompt 10 - Claude - Final GitHub readiness check

**Purpose:** Make sure the project is understandable and reproducible for course submission.

**Prompt:**
Act as a GitHub repository reviewer. Review the folder structure for a Streamlit prototype that uses synthetic data and a Jupyter notebook. Check whether the README explains installation, how to run the app, where the synthetic data are generated, how to configure Gemini secrets, what files should not be committed, and how the prototype connects to the dissertation RQ2 chatbot goal. Suggest any missing files or unclear instructions.

**Validation:** I confirmed that the package includes `streamlit_app.py`, `src/team_dynamics_metrics.py`, an executed notebook, an identity file, a prompt log, generated synthetic outputs, `requirements.txt`, `.gitignore`, `.streamlit/secrets.toml.example`, and a README.
