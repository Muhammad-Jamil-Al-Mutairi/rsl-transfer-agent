# ⚽ Roshn Saudi League (RSL) — Transfer Market & FFP Advisor Agent

An Advanced Agentic RAG (Retrieval-Augmented Generation) infrastructure built for the **SDAIA Applied Generative AI** Capstone Project. This production-ready system interprets complex regulatory guidelines, tracks domestic squad constraints, computes live financial metrics, and scores media transfer rumors for the 2026/27 Roshn Saudi League (RSL) season.

---

## 🏗️ System Architecture Overview

The system transitions past standard vector lookups by executing an **Agentic Loop** powered by Gemini. When a user asks a complex multi-part compliance question, the agent builds a sequential plan, routes between vector databases and local programmatic logic engines, and merges data outputs into a single, fully grounded answer with complete page-level citations.

![System Architecture Diagram for RSL Transfer Advisor](https://github.com/user-attachments/assets/5e664da0-ef4a-4730-af7f-110161ab66ee)

### 🗲 Operational Workflow Topology
1. **User Request Processing:** User queries are received via the Streamlit front-end context.
2. **Chain-of-Thought Orchestration:** The Gemini Core reasons through the prompt to discover if structural parameters (like conversion rates, squad databases, or specific rules pages) are needed.
3. **Execution Loop & Tool Integration:** * **Semantic Search:** Fetches text nodes from the Qdrant Cloud Cluster.
   * **Programmatic Engine:** Coordinates calculations through 7 custom Python modules.
4. **Answer Synthesis & Evaluation:** Merges the tool payloads, validates that rules aren't violated, and renders text strings with explicit, un-hallucinated source files.

---

## 🛠️ Data Pipeline & Ingestion Infrastructure

### 1. Hybrid Search Architecture
To prevent the omissions common in standard vector lookups, this repository implements a dual **Dense Vector + Sparse Keyword** retrieval pipeline:
* **Dense Semantic Vector Engine:** Utilizes Google’s `text-embedding-004` (768-dimensional model) to capture conceptual inquiries regarding regulations, exceptions, and eligibility definitions.
* **Sparse Lexical Engine:** Integrates a local `rank-bm25` pipeline to track absolute string matches like specific player names, transaction codes, or unique regulatory acronyms ("FFP", "FSP").
* **Reciprocal Rank Fusion (RRF):** Blends matching items into a single, highly relevant ranking context block.

### 2. Ingestion Stages
* **Parsing:** Chunks regulatory documents by strict logical page numbers.
* **Seeding:** Generates index blocks and registers payloads inside a serverless **Qdrant Cloud Cluster** hosted on AWS/GCP.

---

## 🧰 Specialized Python Tool Registry

The agent handles complex reasoning by routing arguments out to a set of 7 discrete programmatic tools:

| Category | Tool Name | Scope & Technical Functionality |
| :--- | :--- | :--- |
| **Rumor Intelligence** | `calculate_transfer_score` | Computes a probability metric (0–100%) by scoring reporting tiers, contractual status, and team rivalries. |
| | `filter_latest_updates` | Segregates temporal real-time reports based on source validity dates. |
| **Squad Analytics** | `check_squad_registration` | Queries JSON data to calculate team quota balances against the maximum limit of 8 senior foreign slots. |
| | `filter_under21_foreign_slot` | Validates birthdate bounds (born on/after Jan 1, 2005) to check eligibility for the 2 dedicated U21 development slots. |
| | `check_homegrown_player_ratio` | Validates compliance against domestic squad ratio guidelines. |
| **Financial Engine** | `currency_converter_saudi_riyal` | Normalizes standard currencies against the official reference exchange rate (1 EUR = 4.08 SAR) and flags FFP violations. |
| **Comparative Core**| `compare_player_stats` | Compares performance data metrics between targets and current squad members. |

---

## 🏆 Evaluation: The Golden Set

To ensure absolute accuracy and prevent LLM hallucinations, this agent was benchmarked against a domain-specific **Golden Set**. This dataset evaluates the system across tool routing accuracy, retrieval precision, chain-of-thought execution, and citation fidelity.

| Evaluation Category | Question | Expected Tool Call | Expected Result |
| :--- | :--- | :--- | :--- |
| **Tool Isolation** | Calculate the transfer score for a Tier 1 rumor with agreed terms and agreed fee, but between rival clubs. | `calculate_transfer_score` | 80% (Very Likely) |
| **Data Registry Lookup** | Is Al-Hilal currently compliant with the senior foreign player quota? | `check_squad_registration` | Non-compliant (9 players, exceeding limit of 8) |
| **Financial Logic** | Convert €45M to Saudi Riyal and check the transfer fee against standard FFP thresholds. | `currency_converter_saudi_riyal` | 183.6M SAR, exceeds 75M SAR FSP threshold |
| **Multi-Tool Pipeline** | A Tier 2 source reports Al-Nassr agreed to terms to sign a foreign forward born in 2006 for €20M. Run a full diagnostic. | `calculate_transfer_score` -> `filter_under21_foreign_slot` -> `currency_converter_saudi_riyal` | Score: 85%, U21 Eligible, 81.6M SAR |
| **Dossier Retrieval Grounding** | What are the exact squad registration rules and age cutoffs for foreign players in the 2026 season? | `Qdrant Hybrid Search (RAG)` | Max 8 senior, 2 U21 (born on/after Jan 1, 2005) + Page citations |

---

## 🚀 Production Deployment & Installation

### 1. Local Configuration Setup
To initialize and test the repository on your machine:

```powershell
# Clone the project directory
git clone [https://github.com/Muhammad-Jamil-Al-Mutairi/rsl-transfer-agent.git](https://github.com/Muhammad-Jamil-Al-Mutairi/rsl-transfer-agent.git)
cd rsl-transfer-agent

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the project locally
streamlit run app.py

# ⚽ Roshn Saudi League (RSL) — Transfer Market & FFP Advisor Agent

[![Live Demo](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://rsl-transfer-agent-efamanjsztajcbtspqfcnx.streamlit.app/)

**Try the live agent here:** [https://rsl-transfer-agent-efamanjsztajcbtspqfcnx.streamlit.app/](https://rsl-transfer-agent-efamanjsztajcbtspqfcnx.streamlit.app/)

An Advanced Agentic RAG (Retrieval-Augmented Generation) infrastructure built for the **SDAIA Applied Generative AI** Capstone Project...