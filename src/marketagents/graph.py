"""Graf agentów w LangGraph.

dane -> analityk techniczny -> redaktor -> weryfikator --(błędy)--> redaktor
                                                       \\--(zgodne)--> koniec
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from marketagents import prompts
from marketagents.facts import Fact, FactSheet
from marketagents.verifier import mark_unverified, verify

MAX_FIXES = 2


class ReportState(TypedDict, total=False):
    ticker: str
    facts: list[dict]
    analyses: Annotated[dict[str, str], operator.or_]
    draft: str
    issues: list[str]
    unverified: list[str]
    attempts: int
    report: str
    verified: bool


def _text(message) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def build_graph(llm: BaseChatModel, load_facts: Callable[[str], list[Fact]]):
    def load_data(state: ReportState) -> ReportState:
        return {"facts": FactSheet(load_facts(state["ticker"])).to_records(), "attempts": 0}

    def technical_analyst(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        prompt = prompts.TECHNICAL_ANALYST.format(
            ticker=state["ticker"], rules=prompts.PLACEHOLDER_RULES, facts=facts.as_prompt_table()
        )
        return {"analyses": {"techniczna": _text(llm.invoke(prompt))}}

    def editor(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        prompt = prompts.EDITOR.format(
            ticker=state["ticker"],
            rules=prompts.PLACEHOLDER_RULES,
            facts=facts.as_prompt_table(),
            analyses="\n\n".join(f"### {k}\n{v}" for k, v in state["analyses"].items()),
        )
        if state.get("issues"):
            prompt += "\n\n" + prompts.EDITOR_FIX.format(
                issues="\n".join(f"- {i}" for i in state["issues"]), draft=state["draft"]
            )
        return {"draft": _text(llm.invoke(prompt))}

    def verifier(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        result = verify(state["draft"], facts)
        return {
            "issues": result.issues(),
            "unverified": result.unverified_numbers,
            "attempts": state.get("attempts", 0) + 1,
            "report": result.rendered,
            "verified": result.ok,
        }

    def finalize(state: ReportState) -> ReportState:
        if state["verified"]:
            return {}
        return {"report": mark_unverified(state["report"], state["unverified"])}

    def after_verification(state: ReportState) -> str:
        if state["verified"] or state["attempts"] > MAX_FIXES:
            return "finalize"
        return "editor"

    graph = StateGraph(ReportState)
    graph.add_node("load_data", load_data)
    graph.add_node("technical_analyst", technical_analyst)
    graph.add_node("editor", editor)
    graph.add_node("verifier", verifier)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "technical_analyst")
    graph.add_edge("technical_analyst", "editor")
    graph.add_edge("editor", "verifier")
    graph.add_conditional_edges("verifier", after_verification, ["editor", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()
