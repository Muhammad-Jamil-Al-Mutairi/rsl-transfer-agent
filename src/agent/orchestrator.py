"""
Gemini-powered agent orchestrator for the RSL Transfer Market & FFP Advisor.

Responsibilities:
  1. Retrieve grounding context chunks from the Qdrant Hybrid RAG store.
  2. Send the user prompt + retrieved context + the 7 registered tools to
     Gemini.
  3. Run a manual function-calling loop: whenever Gemini emits a
     function_call, execute the matching tool in `rsl_tools.py`, feed the
     result back as a function_response turn, and continue until Gemini
     produces a final text answer.
  4. Return the final answer together with a structured tool-execution
     trace and RAG citations, so the Streamlit UI can render both.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.rag.vector_store import search_documents
from src.tools.rsl_tools import TOOL_REGISTRY

load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_TOOL_ITERATIONS = 6

SYSTEM_INSTRUCTION = """\
You are the Roshn Saudi League (RSL) Transfer Market & FFP Advisor Agent — an
expert assistant for club sporting directors and analysts covering the 2026
summer transfer window.

You have two sources of ground truth:

1. RETRIEVED DOSSIER CONTEXT — excerpts from the official RSL transfer
   dossier, supplied to you below the user's question, each tagged with its
   exact source filename and page number, e.g.
   [Source: rsl_transfer_dossier_2026.txt | Page 3: FFP Overview & Salary Cap Mechanism]
2. LIVE TOOLS — 7 Python functions for math, squad-quota validation, salary
   conversion, and rumor filtering. Call a tool whenever the question needs
   a calculation, a quota/eligibility check, a currency conversion, a stat
   comparison, or rumor filtering/scoring — never do that arithmetic yourself.

STRICT CITATION RULE: whenever you state a fact that comes from the
retrieved dossier context, you MUST cite it inline in the exact format
(Source: <filename>, Page <N>) immediately after the sentence that uses it.
Never state a regulatory rule, transfer fact, rumor, or statistic from the
dossier without this citation. If the retrieved context does not contain
the answer, say so plainly instead of guessing — do not invent citations.

When you use a tool, briefly state the tool's result in plain language as
part of your answer (the UI will separately show the full tool trace), but
tool results themselves don't need page citations since they are live
calculations, not dossier facts.

Be concise, precise, and structure multi-part answers with short headers or
bullet points when useful. Always answer as an expert advisor, not a generic
chatbot.
"""


def _schema(type_: str, description: str = "", **kwargs: Any) -> types.Schema:
    return types.Schema(type=type_, description=description, **kwargs)


def build_tools() -> types.Tool:
    """Build the Gemini FunctionDeclaration set for all 7 RSL tools."""
    declarations = [
        types.FunctionDeclaration(
            name="calculate_transfer_score",
            description=(
                "Compute a 0-100 likelihood score for a transfer rumor from source "
                "reliability tier, whether personal terms/fee are agreed, and "
                "whether the same player is linked to a direct rival club."
            ),
            parameters=_schema(
                types.Type.OBJECT,
                properties={
                    "source_tier": _schema(
                        types.Type.INTEGER,
                        "Source reliability tier: 1 (most reliable), 2, or 3 (least reliable).",
                    ),
                    "personal_terms_agreed": _schema(
                        types.Type.BOOLEAN, "Whether personal terms are reported agreed."
                    ),
                    "fee_agreed": _schema(
                        types.Type.BOOLEAN, "Whether a transfer fee is reported agreed between clubs."
                    ),
                    "is_rival_transfer": _schema(
                        types.Type.BOOLEAN,
                        "Whether the player is also strongly linked to a direct league rival (default false).",
                    ),
                },
                required=["source_tier", "personal_terms_agreed", "fee_agreed"],
            ),
        ),
        types.FunctionDeclaration(
            name="filter_latest_updates",
            description=(
                "Filter a list of transfer rumor/update objects down to only those "
                "within max_age_hours, sorted newest first. Each update should have "
                "an 'age_hours' number or an ISO 'timestamp' string."
            ),
            parameters=_schema(
                types.Type.OBJECT,
                properties={
                    "updates": _schema(
                        types.Type.ARRAY,
                        "List of update/rumor objects to filter.",
                        items=_schema(types.Type.OBJECT, "A single rumor/update record."),
                    ),
                    "max_age_hours": _schema(
                        types.Type.INTEGER, "Maximum age in hours to keep an update (default 48)."
                    ),
                },
                required=["updates"],
            ),
        ),
        types.FunctionDeclaration(
            name="check_squad_registration",
            description=(
                "Validate whether a club can register a new signing under the RSL "
                "'8+2' foreign player quota rule (max 8 senior foreign + 2 U21 "
                "foreign development slots). Reads live squad data for the club."
            ),
            parameters=_schema(
                types.Type.OBJECT,
                properties={
                    "club_name": _schema(types.Type.STRING, "RSL club name, e.g. 'Al-Nassr'."),
                    "is_foreign": _schema(types.Type.BOOLEAN, "Whether the incoming player is a foreign national."),
                    "birth_year": _schema(types.Type.INTEGER, "Player's birth year."),
                },
                required=["club_name", "is_foreign", "birth_year"],
            ),
        ),
        types.FunctionDeclaration(
            name="filter_under21_foreign_slot",
            description=(
                "Determine whether a foreign player qualifies for one of the two "
                "dedicated U21 foreign development slots (born on/after 2005)."
            ),
            parameters=_schema(
                types.Type.OBJECT,
                properties={"birth_year": _schema(types.Type.INTEGER, "Player's birth year.")},
                required=["birth_year"],
            ),
        ),
        types.FunctionDeclaration(
            name="check_homegrown_player_ratio",
            description=(
                "Check whether a club's registered squad meets the RSL 50% minimum "
                "Saudi national ('homegrown') player guideline."
            ),
            parameters=_schema(
                types.Type.OBJECT,
                properties={"club_name": _schema(types.Type.STRING, "RSL club name, e.g. 'Al-Hilal'.")},
                required=["club_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="currency_converter_saudi_riyal",
            description="Convert an amount in EUR, GBP, or USD to Saudi Riyal (SAR) using RSL reference rates.",
            parameters=_schema(
                types.Type.OBJECT,
                properties={
                    "amount": _schema(types.Type.NUMBER, "Amount in the source currency (non-negative)."),
                    "from_currency": _schema(
                        types.Type.STRING,
                        "Source currency code.",
                        enum=["EUR", "GBP", "USD"],
                    ),
                },
                required=["amount", "from_currency"],
            ),
        ),
        types.FunctionDeclaration(
            name="compare_player_stats",
            description=(
                "Compute a side-by-side statistical comparison (goals, assists, key "
                "passes per 90, minutes per goal) between a target signing and an "
                "existing squad player."
            ),
            parameters=_schema(
                types.Type.OBJECT,
                properties={
                    "target_player_metrics": _schema(
                        types.Type.OBJECT,
                        "Metrics for the prospective signing, e.g. {goals, assists, "
                        "key_passes_per_90, minutes_played}.",
                    ),
                    "current_player_metrics": _schema(
                        types.Type.OBJECT,
                        "Metrics for the existing squad player, same shape as target_player_metrics.",
                    ),
                },
                required=["target_player_metrics", "current_player_metrics"],
            ),
        ),
    ]
    return types.Tool(function_declarations=declarations)


def _format_context_block(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""
    lines = ["RETRIEVED DOSSIER CONTEXT:"]
    for chunk in chunks:
        header = f"[Source: {chunk['source']} | Page {chunk['page']}: {chunk.get('page_title', '')}] (similarity={chunk['score']})"
        lines.append(header)
        lines.append(chunk["text"])
        lines.append("---")
    return "\n".join(lines)


class RSLAgentOrchestrator:
    """Stateless-per-call orchestrator wrapping Gemini function calling + RAG."""

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
        self.client = genai.Client(api_key=api_key)
        self.model = model or DEFAULT_MODEL
        self.tools = build_tools()

    def answer(
        self,
        user_query: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int = 4,
    ) -> dict[str, Any]:
        """Answer a user query using RAG context + tool-calling.

        Args:
            user_query: The latest user message.
            chat_history: Prior turns as [{"role": "user"|"assistant", "content": str}, ...].
            top_k: Number of RAG chunks to retrieve.

        Returns:
            A dict with keys: "answer" (str), "citations" (list of retrieved
            chunk dicts), "tool_trace" (list of {tool, args, result} dicts),
            and "rag_error" (str | None).
        """
        rag_result = search_documents(user_query, top_k=top_k)
        context_chunks = rag_result.get("results", []) if rag_result.get("success") else []
        rag_error = None if rag_result.get("success") else rag_result.get("error")

        contents: list[types.Content] = []
        for turn in chat_history or []:
            role = "model" if turn.get("role") == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=turn.get("content", ""))]))

        context_block = _format_context_block(context_chunks)
        user_turn_text = f"{context_block}\n\nUSER QUESTION: {user_query}" if context_block else user_query
        contents.append(types.Content(role="user", parts=[types.Part(text=user_turn_text)]))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[self.tools],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )

        tool_trace: list[dict[str, Any]] = []
        final_text = ""

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
            if not response.candidates:
                break
            model_content = response.candidates[0].content
            contents.append(model_content)

            function_calls = [
                part.function_call for part in (model_content.parts or []) if part.function_call
            ]
            if not function_calls:
                final_text = response.text or ""
                break

            response_parts: list[types.Part] = []
            for fc in function_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                tool_fn = TOOL_REGISTRY.get(tool_name)
                if tool_fn is None:
                    result = {"success": False, "error": f"Unknown tool '{tool_name}'."}
                else:
                    try:
                        result = tool_fn(**tool_args)
                    except Exception as exc:  # tool must never crash the agent loop
                        result = {"success": False, "error": str(exc)}
                tool_trace.append({"tool": tool_name, "args": tool_args, "result": result})
                response_parts.append(types.Part.from_function_response(name=tool_name, response=result))

            contents.append(types.Content(role="user", parts=response_parts))
        else:
            final_text = final_text or (
                "I reached the tool-call budget for this turn without finishing. "
                "Please rephrase or break the question into smaller parts."
            )

        if not final_text:
            final_text = "I couldn't generate a response for that question."

        return {
            "answer": final_text,
            "citations": context_chunks,
            "tool_trace": tool_trace,
            "rag_error": rag_error,
        }
