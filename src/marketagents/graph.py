"""Graf agentów w LangGraph.

                 /-> analityk techniczny     -\
dane -> fakty --+--> analityk fundamentalny --+-> redaktor -> weryfikator --(zgodne)--> koniec
                 +-> analityk newsów         -+       ^              |
                 \\-> analityk makro          -/       +--(błędy)-----+

Analitycy działają równolegle; analityk, dla którego zabrakło danych, jest pomijany.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from marketagents import prompts
from marketagents.facts import FUNDAMENTAL, MACRO, TECHNICAL, Fact, FactSheet
from marketagents.llm import message_text
from marketagents.verifier import mark_unverified, verify

MAX_FIXES = 2
MAX_NEWS = 15


@dataclass
class MarketData:
    facts: list[Fact]
    news: list[dict] = field(default_factory=list)


class ReportState(TypedDict, total=False):
    ticker: str
    facts: list[dict]
    news: list[dict]
    analyses: Annotated[dict[str, str], operator.or_]
    draft: str
    issues: list[str]
    unverified: list[str]
    attempts: int
    report: str
    verified: bool


def _format_news(news: list[dict]) -> str:
    return "\n".join(
        f"- [{n['published'][:10]}] {n['source']}: {n['headline']}. {n.get('summary', '')}".strip()
        for n in news[:MAX_NEWS]
    )


def build_graph(llm: BaseChatModel, load_data: Callable[[str], MarketData]):
    def load(state: ReportState) -> ReportState:
        data = load_data(state["ticker"])
        return {
            "facts": FactSheet(data.facts).to_records(),
            "news": data.news,
            "attempts": 0,
        }

    def analyst(name: str, template: str, group: str, needs_news: bool = False):
        def node(state: ReportState) -> ReportState:
            facts = FactSheet.from_records(state["facts"]).group(group)
            has_data = any(f.group == group for f in facts)
            if needs_news and not state.get("news"):
                return {}
            if not needs_news and not has_data:
                return {}
            prompt = template.format(
                ticker=state["ticker"],
                rules=prompts.PLACEHOLDER_RULES,
                facts=facts.as_prompt_table(),
                news=_format_news(state.get("news", [])),
            )
            return {"analyses": {name: message_text(llm.invoke(prompt))}}

        return node

    def editor(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        analyses = state.get("analyses", {})
        prompt = prompts.EDITOR.format(
            ticker=state["ticker"],
            rules=prompts.PLACEHOLDER_RULES,
            facts=facts.as_prompt_table(),
            analyses="\n\n".join(f"### Analiza: {k}\n{v}" for k, v in analyses.items()),
        )
        if state.get("issues"):
            prompt += "\n\n" + prompts.EDITOR_FIX.format(
                issues="\n".join(f"- {i}" for i in state["issues"]), draft=state["draft"]
            )
        return {"draft": message_text(llm.invoke(prompt))}

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

    analysts = {
        "technical_analyst": analyst("techniczna", prompts.TECHNICAL_ANALYST, TECHNICAL),
        "fundamental_analyst": analyst("fundamentalna", prompts.FUNDAMENTAL_ANALYST, FUNDAMENTAL),
        "news_analyst": analyst("wiadomości", prompts.NEWS_ANALYST, "", needs_news=True),
        "macro_analyst": analyst("makroekonomiczna", prompts.MACRO_ANALYST, MACRO),
    }

    graph = StateGraph(ReportState)
    graph.add_node("load_data", load)
    graph.add_node("editor", editor)
    graph.add_node("verifier", verifier)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "load_data")
    for node_name, node in analysts.items():
        graph.add_node(node_name, node)
        graph.add_edge("load_data", node_name)
    graph.add_edge(list(analysts), "editor")
    graph.add_edge("editor", "verifier")
    graph.add_conditional_edges("verifier", after_verification, ["editor", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


def run_report(graph, ticker: str, on_step: Callable[[str], None] | None = None) -> ReportState:
    """Uruchamia graf etap po etapie; on_step dostaje nazwę każdego zakończonego węzła."""
    state: ReportState = {"ticker": ticker}
    for update in graph.stream(state, stream_mode="updates"):
        for node, values in update.items():
            for key, value in (values or {}).items():
                if key == "analyses":
                    state.setdefault("analyses", {}).update(value)
                else:
                    state[key] = value
            if on_step:
                on_step(node)
    return state
