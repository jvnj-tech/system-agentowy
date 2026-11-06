"""Graf agentów w LangGraph.

                 /-> analityk techniczny     -\
dane -> fakty --+--> analityk fundamentalny --+-> redaktor -> weryfikator --(zgodne)--> przegląd
                 +-> analityk newsów         -+       ^              |                 redaktora
                 \\-> analityk makro          -/       +--(błędy)-----+

Analitycy działają równolegle; analityk, dla którego zabrakło danych, jest pomijany.
Weryfikator sprawdza raport złożony przez redaktora, a błędy poprawia redaktor (najwyżej
MAX_FIXES razy). Po pozytywnej weryfikacji redaktor jeden raz przegląda raport, a poprawiony
raport ponownie sprawdza weryfikator. Analityk wiadomości dostaje ponumerowaną listę newsów
o spółce i zmiany kursu z karty, a każde zdanie o wiadomościach musi wskazywać numer źródła.
Opis zakresu danych w sekcji ograniczeń dopisuje kod.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from marketagents import prompts
from marketagents.coverage import coverage_note
from marketagents.facts import FUNDAMENTAL, MACRO, TECHNICAL, Fact, FactSheet
from marketagents.interpretation import mark_sentences, prompt_table
from marketagents.llm import AllModelsExhausted, message_text
from marketagents.verifier import SOURCE_MARKER, cite, mark_unverified, render, verify

MAX_FIXES = 3
MAX_NEWS = 15
# Ostatnia sekcja raportu; jej brak oznacza, że model urwał odpowiedź.
LAST_SECTION = "## Ograniczenia analizy"
# Sekcje raportu w wymaganej kolejności i analiza, z której powstaje każda z nich
# (sekcja bez analizy cząstkowej, np. bez danych makro, jest pomijana).
SECTIONS = [
    ("## Podsumowanie", None),
    ("## Analiza techniczna", "techniczna"),
    ("## Analiza fundamentalna", "fundamentalna"),
    ("## Wiadomości", "wiadomości"),
    ("## Otoczenie makroekonomiczne", "makroekonomiczna"),
    (LAST_SECTION, None),
]


def expected_sections(analyses: dict[str, str]) -> list[tuple[str, str | None]]:
    return [(title, name) for title, name in SECTIONS if name is None or analyses.get(name)]


def section_issues(draft: str, analyses: dict[str, str]) -> list[str]:
    """Brakujące sekcje i sekcje w złej kolejności (model potrafi zgubić sekcję przy poprawce)."""
    expected = expected_sections(analyses)
    lines = [line.strip() for line in draft.splitlines()]
    positions = {title: lines.index(title) for title, _ in expected if title in lines}
    issues = [
        f"Raport jest niekompletny: brak sekcji „{title}”"
        for title, _ in expected
        if title not in positions
    ]
    present = [title for title, _ in expected if title in positions]
    if [positions[t] for t in present] != sorted(positions[t] for t in present):
        order = ", ".join(f"„{t}”" for t, _ in expected)
        issues.append(f"Sekcje raportu są w złej kolejności; wymagana kolejność: {order}")
    return issues


# Fakty, które dostaje analityk wiadomości, by porównać opinie z artykułów z danymi.
NEWS_CONTEXT = ("change_1m", "change_3m", "change_1y")


@dataclass
class MarketData:
    facts: list[Fact]
    news: list[dict] = field(default_factory=list)
    # Nazwa spółki: jej wymienienie w zdaniu o wiadomościach nie wymaga potwierdzenia w źródle.
    company: str | None = None


class ReportState(TypedDict, total=False):
    ticker: str
    company: str
    facts: list[dict]
    news: list[dict]
    analyses: Annotated[dict[str, str], operator.or_]
    draft: str
    issues: list[str]
    # Liczba problemów znalezionych przez weryfikator w kolejnych wersjach raportu.
    issue_counts: Annotated[list[int], operator.add]
    unverified: list[str]
    flagged: list[str]
    uncited: list[str]
    attempts: int
    report: str
    report_html: str
    verified: bool
    # Model, który napisał raport, i liczba rozpoczęć od nowa po wyczerpaniu limitu.
    model: str
    restarts: int
    # Sprawdzenie sekcji przez redaktora: czy się odbyło i jakie uwagi zgłosił.
    reviewed: bool
    review_issues: list[str]


def _format_news(news: list[dict]) -> str:
    lines = []
    for i, n in enumerate(news[:MAX_NEWS], start=1):
        text = f"[{i}] {n['published'][:10]}, {n['source']}: {n['headline']}."
        lines.append(f"{text} {n.get('summary') or ''}".strip())
    return "\n".join(lines)


# Brak treści w odpowiedzi modelu (zdarza się przy długich poleceniach, gdy model zużyje
# odpowiedź na rozumowanie) jest ponawiany tyle razy, zanim zostanie uznany za brak analizy.
EMPTY_RETRIES = 1
EMPTY_REPORT = (
    "Model językowy nie zwrócił treści raportu. Spróbuj wygenerować raport ponownie "
    "lub wybierz innego dostawcę modelu."
)


REVIEW_NONE = "BRAK UWAG"
REVIEW_ITEM = re.compile(r"^\s*[-*\d.)]+\s*(.+)$")


def parse_review(text: str) -> list[str]:
    """Uwagi redaktora: wypunktowane wiersze; „BRAK UWAG” oznacza pustą listę."""
    if REVIEW_NONE in text.upper() and not REVIEW_ITEM.search(text.split(REVIEW_NONE)[0]):
        return []
    return [m.group(1).strip() for line in text.splitlines() if (m := REVIEW_ITEM.match(line))]


def _compact_news(news: list[dict]) -> str:
    """Lista wiadomości do sprawdzenia przez redaktora: nagłówek i początek streszczenia."""
    return "\n".join(
        f"[{i}] {n.get('source', '')}: {n.get('headline', '')}. {str(n.get('summary') or '')[:300]}"
        for i, n in enumerate(news[:MAX_NEWS], start=1)
    )


def build_graph(llm: BaseChatModel, load_data: Callable[[str], MarketData], review: bool = True):
    """review: po pozytywnej weryfikacji kodem redaktor sprawdza raport sekcja po sekcji
    (sprzeczności, twierdzenia bez pokrycia w danych, opinie podane jako fakty) i jeden raz
    poprawia zgłoszone miejsca. To ocena modelu, więc nie zmienia statusu weryfikacji."""

    def ask(prompt: str) -> str:
        text = ""
        for _ in range(1 + EMPTY_RETRIES):
            text = message_text(llm.invoke(prompt)).strip()
            if text:
                break
        return text

    def load(state: ReportState) -> ReportState:
        data = load_data(state["ticker"])
        return {
            "facts": FactSheet(data.facts).to_records(),
            "news": data.news[:MAX_NEWS],
            "company": data.company or "",
            "attempts": 0,
        }

    def analyst(name: str, template: str, group: str, needs_news: bool = False):
        def node(state: ReportState) -> ReportState:
            sheet = FactSheet.from_records(state["facts"])
            facts = sheet.group(group)
            has_data = any(f.group == group for f in facts)
            if needs_news:
                facts = FactSheet([*facts, *(sheet[i] for i in NEWS_CONTEXT if i in sheet)])
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
            text = ask(prompt)
            if not text:
                return {}
            return {"analyses": {name: text}}

        return node

    def editor(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        analyses = state.get("analyses", {})
        if state.get("issues"):
            # Do poprawki trafiają tylko analizy sekcji, których brakuje w raporcie: pełny
            # zestaw analiz nie mieści się w limicie tokenów na minutę darmowych planów.
            lines = {line.strip() for line in state["draft"].splitlines()}
            missing = [
                f"### Analiza: {name}\n{analyses[name]}"
                for title, name in expected_sections(analyses)
                if name and title not in lines
            ]
            prompt = prompts.EDITOR_FIX.format(
                ticker=state["ticker"],
                issues="\n".join(f"- {i}" for i in state["issues"]),
                rules=prompts.PLACEHOLDER_RULES,
                facts=prompt_table(facts),
                draft=state["draft"],
            )
            if missing:
                prompt += "\n\nAnalizy cząstkowe brakujących sekcji:\n" + "\n\n".join(missing)
            # Model bez treści w odpowiedzi nie może skasować raportu: zostaje poprzednia wersja.
            return {"draft": ask(prompt) or state["draft"]}
        prompt = prompts.EDITOR.format(
            ticker=state["ticker"],
            rules=prompts.PLACEHOLDER_RULES,
            facts=prompt_table(facts),
            analyses="\n\n".join(f"### Analiza: {k}\n{v}" for k, v in analyses.items()),
        )
        return {"draft": ask(prompt)}

    def verifier(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        has_news = bool(state.get("analyses", {}).get("wiadomości"))
        result = verify(
            state["draft"],
            facts,
            news=state.get("news", []) if has_news else [],
            names=(state["ticker"], state.get("company", "")),
        )
        issues = result.issues() + section_issues(state["draft"], state.get("analyses", {}))
        return {
            "issues": issues,
            "issue_counts": [len(issues)],
            "unverified": result.to_mark,
            "flagged": result.flagged_sentences,
            "uncited": result.uncited,
            "attempts": state.get("attempts", 0) + 1,
            "report": result.rendered,
            "verified": not issues,
        }

    def finalize(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        news = state.get("news", [])
        draft = state["draft"] or EMPTY_REPORT
        if not state["verified"]:
            draft = mark_sentences(draft, state.get("flagged", []))
            draft = mark_sentences(draft, state.get("uncited", []), SOURCE_MARKER)
            # Oznaczane są liczby wpisane przez model, zanim w tekst trafią wartości z karty:
            # inaczej oznaczenie trafiłoby też do poprawnej wartości o tym samym zapisie.
            draft = mark_unverified(draft, state["unverified"])
        # Zakres danych dopisuje kod po weryfikacji, więc jego daty nie są sprawdzane
        # jak tekst modelu.
        note = "\n\n" + coverage_note(facts, news) + "\n"
        plain, html = render(draft, facts), render(draft, facts, html=True)
        return {
            "report": cite(plain, news).rstrip() + note,
            "report_html": cite(html, news).rstrip() + note,
        }

    def editor_review(state: ReportState) -> ReportState:
        facts = FactSheet.from_records(state["facts"])
        prompt = prompts.EDITOR_REVIEW.format(
            ticker=state["ticker"],
            facts=prompt_table(facts),
            news=_compact_news(state.get("news", [])) or "(brak wiadomości)",
            draft=state["draft"],
        )
        found = parse_review(ask(prompt))
        if not found:
            return {"reviewed": True, "review_issues": []}
        issues = [f"Redaktor (sprawdzenie sekcji): {i}" for i in found]
        return {"reviewed": True, "review_issues": found, "issues": issues, "verified": False}

    def after_verification(state: ReportState) -> str:
        if state["verified"] and review and not state.get("reviewed"):
            return "editor_review"
        # Poprawka po uwagach redaktora nie zmniejsza limitu poprawek błędów weryfikatora.
        fixes = state["attempts"] - 1 - int(bool(state.get("review_issues")))
        if state["verified"] or fixes >= MAX_FIXES:
            return "finalize"
        return "editor"

    def after_review(state: ReportState) -> str:
        return "editor" if state.get("review_issues") else "finalize"

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
    graph.add_node("editor_review", editor_review)
    graph.add_conditional_edges(
        "verifier", after_verification, ["editor", "editor_review", "finalize"]
    )
    graph.add_conditional_edges("editor_review", after_review, ["editor", "finalize"])
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
                else:
                    state[key] = value
            if on_step:
                on_step(node)
    return state


def run_report_with_fallback(
    llm: BaseChatModel,
    make_graph: Callable[[BaseChatModel], object],
    ticker: str,
    on_step: Callable[[str], None] | None = None,
    config: dict | None = None,
    on_restart: Callable[[str], None] | None = None,
) -> ReportState:
    """Raport od początku do końca pisze jeden model z puli.

    Gdy model wyczerpie limit dobowy (albo serwer jest przeciążony), raport zaczyna się od
    nowa na następnym dostępnym modelu, żeby nie składać go z tekstów różnych modeli.
    Limit minutowy jest odczekiwany na tym samym modelu (w puli). on_restart dostaje nazwę
    modelu, na którym przerwano raport.
    """
    if not hasattr(llm, "only"):
        return run_report(make_graph(llm), ticker, on_step, config)
    tried: list[str] = []
    while True:
        candidates = [s for s in llm.available() if s not in tried]
        if not candidates:
            raise llm.exhausted(f"{s}: limit" for s in tried)
        spec = candidates[0]
        tried.append(spec)
        try:
            state = run_report(make_graph(llm.only(spec)), ticker, on_step, config)
        except AllModelsExhausted:
            if on_restart:
                on_restart(spec)
            continue
        state["model"] = spec
        state["restarts"] = len(tried) - 1
        return state
