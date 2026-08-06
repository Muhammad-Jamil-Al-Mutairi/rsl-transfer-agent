"""
Streamlit dashboard for the Roshn Saudi League (RSL) Transfer Market & FFP
Advisor Agent — Hybrid RAG (Qdrant + Gemini embeddings) grounded chat backed
by 7 dynamic Python tools for quota validation, currency conversion, rumor
scoring, and stat comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.agent.orchestrator import RSLAgentOrchestrator
from src.rag import vector_store
from src.tools.rsl_tools import check_squad_registration, currency_converter_saudi_riyal

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
SQUADS_PATH = PROJECT_ROOT / "data" / "squads.json"

st.set_page_config(
    page_title="RSL Transfer Market & FFP Advisor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUGGESTED_PROMPTS = [
    "Can Al-Nassr register a 20-year-old foreign winger?",
    "Calculate the transfer score for a Tier 1 rumor with agreed terms and fee, no rivalry.",
    "Convert €45M to SAR and check the FFP materiality threshold impact.",
    "Is Al-Hilal currently compliant with the foreign player quota?",
    "Compare the target forward Nikolai Petrov against our current Al-Nassr forward.",
    "Does Al-Ittihad meet the 50% homegrown player ratio guideline?",
]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_orchestrator() -> RSLAgentOrchestrator | None:
    try:
        return RSLAgentOrchestrator()
    except RuntimeError:
        return None


@st.cache_data(show_spinner=False, ttl=20)
def get_system_status() -> tuple[dict, dict]:
    return vector_store.check_gemini_status(), vector_store.check_qdrant_status()


def load_club_names() -> list[str]:
    try:
        with open(SQUADS_PATH, "r", encoding="utf-8") as fh:
            return sorted(json.load(fh).keys())
    except (OSError, json.JSONDecodeError):
        return ["Al-Nassr", "Al-Hilal", "Al-Ittihad", "Al-Ahli"]


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------

def status_badge(label: str, connected: bool, detail: str = "") -> None:
    color = "#1a7f37" if connected else "#cf222e"
    bg = "#1a7f3722" if connected else "#cf222e22"
    icon = "🟢" if connected else "🔴"
    st.markdown(
        f'<div style="padding:6px 10px;border-radius:8px;background:{bg};'
        f'border:1px solid {color};margin-bottom:6px;font-size:0.9rem;">'
        f"{icon} <strong>{label}</strong> {detail}</div>",
        unsafe_allow_html=True,
    )


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**📎 Citations**")
    badges = []
    for c in citations:
        badges.append(
            '<span style="background:#1f6feb22;border:1px solid #1f6feb;color:#1f6feb;'
            "padding:3px 10px;border-radius:12px;font-size:0.78rem;margin-right:6px;"
            f'display:inline-block;margin-bottom:6px;">📄 {c["source"]} · Page {c["page"]}'
            f" · {c.get('page_title', '')} · score {c['score']}</span>"
        )
    st.markdown(" ".join(badges), unsafe_allow_html=True)


def render_tool_trace(tool_trace: list[dict]) -> None:
    for t in tool_trace:
        with st.expander(f"🔧 Tool call: `{t['tool']}`", expanded=False):
            st.caption("Arguments")
            st.json(t["args"])
            st.caption("Result")
            st.json(t["result"])


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚽ RSL Advisor")
    st.caption("Transfer Market & FFP Compliance Agent")

    st.subheader("System Status")
    gemini_status, qdrant_status = get_system_status()
    status_badge("Gemini API", gemini_status.get("connected", False))
    if not gemini_status.get("connected"):
        st.caption(gemini_status.get("error", ""))

    qdrant_detail = ""
    if qdrant_status.get("connected") and qdrant_status.get("collection_exists"):
        qdrant_detail = f"({qdrant_status.get('points_count', 0)} chunks indexed)"
    elif qdrant_status.get("connected"):
        qdrant_detail = "(not indexed yet)"
    status_badge("Qdrant Cloud", qdrant_status.get("connected", False), qdrant_detail)
    if not qdrant_status.get("connected"):
        st.caption(qdrant_status.get("error", ""))

    if st.button("🔄 Refresh status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("📚 Document Ingestion")
    st.caption("Chunks the dossier by page, embeds with `text-embedding-004`, and upserts into Qdrant Cloud.")
    if st.button("🔄 Re-index Transfer Dossier", use_container_width=True, type="primary"):
        with st.spinner("Chunking, embedding, and upserting into Qdrant..."):
            index_result = vector_store.index_documents()
        if index_result.get("success"):
            st.success(
                f"Indexed {index_result['chunks_indexed']} chunks from "
                f"{len(index_result['documents_indexed'])} document(s)."
            )
            st.cache_data.clear()
        else:
            st.error(index_result.get("error", "Indexing failed."))

    st.divider()
    st.subheader("🏟️ Quick Squad Lookup")
    club_names = load_club_names()
    lookup_club = st.selectbox("Club", club_names, key="lookup_club")
    lookup_is_foreign = st.checkbox("Foreign player?", value=True, key="lookup_is_foreign")
    lookup_birth_year = st.number_input(
        "Birth year", min_value=1985, max_value=2026, value=2003, step=1, key="lookup_birth_year"
    )
    if st.button("Check Eligibility", use_container_width=True):
        lookup_result = check_squad_registration(lookup_club, lookup_is_foreign, int(lookup_birth_year))
        if lookup_result.get("success"):
            if lookup_result["eligible"]:
                st.success(f"✅ Eligible — {lookup_result['slot_used']}")
            else:
                st.error("❌ Not eligible")
            st.caption(lookup_result["reason"])
        else:
            st.error(lookup_result.get("error", "Lookup failed."))

    st.divider()
    st.subheader("💱 Currency Converter")
    conv_amount = st.number_input(
        "Amount", min_value=0.0, value=10_000_000.0, step=1_000_000.0, key="conv_amount"
    )
    conv_currency = st.selectbox("From currency", ["EUR", "GBP", "USD"], key="conv_currency")
    if st.button("Convert to SAR", use_container_width=True):
        conv_result = currency_converter_saudi_riyal(conv_amount, conv_currency)
        if conv_result.get("success"):
            st.info(
                f"{conv_amount:,.0f} {conv_currency} = "
                f"**{conv_result['converted_amount_sar']:,.0f} SAR** "
                f"({conv_result['converted_amount_sar_millions']}M)"
            )
        else:
            st.error(conv_result.get("error", "Conversion failed."))

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

st.title("⚽ Roshn Saudi League — Transfer Market & FFP Advisor")
st.caption(
    "Ask about transfer rumors, squad registration quotas, FFP/salary-cap rules, "
    "or player comparisons. Every dossier-sourced fact is cited with its exact page."
)

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    for i, suggestion in enumerate(SUGGESTED_PROMPTS):
        if cols[i % 3].button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.pending_prompt = suggestion

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_citations(message.get("citations", []))
            if message.get("tool_trace"):
                render_tool_trace(message["tool_trace"])

user_prompt = st.chat_input("Ask about transfers, FFP compliance, squad quotas...")
if st.session_state.pending_prompt:
    user_prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if user_prompt:
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt, "citations": [], "tool_trace": []}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    orchestrator = get_orchestrator()

    with st.chat_message("assistant"):
        if orchestrator is None:
            answer_text = (
                "⚠️ Gemini API is not configured. Set `GEMINI_API_KEY` in your `.env` file "
                "(see `.env.example`) and restart the app to chat with the advisor agent. "
                "The sidebar's Quick Squad Lookup and Currency Converter tools work "
                "without an API key."
            )
            st.markdown(answer_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text, "citations": [], "tool_trace": []}
            )
        else:
            history_for_model = [
                {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
            ]

            result = None
            with st.status("Thinking through RSL transfer rules & tools...", expanded=True) as status:
                st.write("🔎 Retrieving relevant dossier pages from Qdrant...")
                try:
                    result = orchestrator.answer(user_prompt, chat_history=history_for_model)
                except Exception as exc:  # noqa: BLE001 - surface any Gemini/Qdrant error to the UI
                    status.update(label="Agent error", state="error")
                    st.error(f"Agent error: {exc}")

                if result is not None:
                    if result.get("rag_error"):
                        st.write(f"⚠️ RAG retrieval issue: {result['rag_error']}")
                    else:
                        st.write(f"✅ Retrieved {len(result['citations'])} dossier chunk(s).")
                    for t in result["tool_trace"]:
                        st.write(f"🔧 Called tool `{t['tool']}` with args `{t['args']}`")
                    status.update(label="Done", state="complete")

            if result is not None:
                st.markdown(result["answer"])
                render_citations(result["citations"])
                if result["tool_trace"]:
                    render_tool_trace(result["tool_trace"])

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "citations": result["citations"],
                        "tool_trace": result["tool_trace"],
                    }
                )
