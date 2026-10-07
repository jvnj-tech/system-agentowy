"""Graf agentów w LangGraph.

                 /-> analityk techniczny     -\
dane -> fakty --+--> analityk fundamentalny --+-> redaktor -> weryfikator --(zgodne)--> koniec
                 +-> analityk newsów         -+       ^              |
                 \\-> analityk makro          -/       +--(błędy)-----+

Analitycy działają równolegle; analityk, dla którego zabrakło danych, jest pomijany.
Każda analiza jest od razu weryfikowana i w razie błędów raz poprawiana przez analityka.
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
from marketagents.interpretation import mark_sentences, prompt_table
from marketagents.llm import message_text
from marketagents.verifier import mark_unverified, render, verify

MAX_FIXES = 2
# Analiza cząstkowa jest sprawdzana od razu i może zostać raz poprawiona przez analityka.
ANALYST_FIXES = 1
MAX_NEWS = 15
# Ostatnia sekcja raportu; jej brak oznacza, że model urwał odpowiedź.
LAST_SECTION = "## Ograniczenia analizy"


@dataclass
class MarketData:
    facts: list[Fact]
    news: list[dict] = field(default_factory=list)


class ReportState(TypedDict, total=False):
    ticker: str
    facts: list[dict]
    news: list[dict]
    analyses: Annotated[dict[str, str], operator.or_]
    analyst_fixes: Annotated[int, operator.add]
    draft: str
    issues: list[str]
    # Liczba problemów znalezionych przez weryfikator w kolejnych wersjach raportu.
    issue_counts: Annotated[list[int], operator.add]
    unverified: list[str]
    flagged: list[str]
    attempts: int
    report: str
    report_html: str
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
                facts=prompt_table(facts),
                news=_format_news(state.get("news", [])),
            )
            text = message_text(llm.invoke(prompt))
            fixes = 0
            while fixes < ANALYST_FIXES:
                result = verify(text, facts)
                if result.ok:
                    break
                issues = "\n".join(f"- {i}" for i in result.issues())
                fix = prompts.ANALYST_FIX.format(issues=issues, draft=text)
                text = message_text(llm.invoke(prompt + "\n\n" + fix))
                fixes += 1
            return {"analyses": {name: text}, "analyst_fixes": fixes}

        return node

    def editor(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        analyses = state.get("analyses", {})
        prompt = prompts.EDITOR.format(
            ticker=state["ticker"],
            rules=prompts.PLACEHOLDER_RULES,
            facts=prompt_table(facts),
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
        issues = result.issues()
        if LAST_SECTION not in state["draft"]:
            issues.append(f"Raport jest niekompletny: brak sekcji „{LAST_SECTION}”")
        return {
            "issues": issues,
            "issue_counts": [len(issues)],
            "unverified": result.to_mark,
            "flagged": result.flagged_sentences,
            "attempts": state.get("attempts", 0) + 1,
            "report": result.rendered,
            "verified": not issues,
        }

    def finalize(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        if state["verified"]:
            return {"report_html": render(state["draft"], facts, html=True)}
        draft = mark_sentences(state["draft"], state.get("flagged", []))
        return {
            "report": mark_unverified(render(draft, facts), state["unverified"]),
            "report_html": mark_unverified(render(draft, facts, html=True), state["unverified"]),
        }

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


def run_report(
    graph,
    ticker: str,
    on_step: Callable[[str], None] | None = None,
    config: dict | None = None,
) -> ReportState:
    """Uruchamia graf etap po etapie; on_step dostaje nazwę każdego zakończonego węzła.

    config trafia do LangGraph, np. z obiektem liczącym tokeny w config["callbacks"].
    Pola z reduktorem w ReportState są łączone tak jak w graph.invoke.
    """
    state: ReportState = {"ticker": ticker}
    for update in graph.stream(state, config=config, stream_mode="updates"):
        for node, values in update.items():
            for key, value in (values or {}).items():
                if key == "analyses":
                    state.setdefault("analyses", {}).update(value)
                elif key == "issue_counts":
                    state.setdefault("issue_counts", []).extend(value)
                elif key == "analyst_fixes":
                    state["analyst_fixes"] = state.get("analyst_fixes", 0) + value
                else:
                    state[key] = value
            if on_step:
                on_step(node)
    return state
