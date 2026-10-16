"""Asystent odpowiadający na pytania o wygenerowany raport.

Obowiązują go te same zasady co autorów raportu: liczby wstawia jako odwołania do karty
faktów, informacje z wiadomości opatruje numerem źródła [n], nie formułuje prognoz,
a odpowiedź przechodzi przez weryfikator (liczby, opisy, prognozy, zgodność nazw
z cytowaną wiadomością). Przy błędach model dostaje jedną szansę na poprawkę, a potem
to, czego nie udało się potwierdzić, jest oznaczane.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from marketagents import prompts
from marketagents.facts import FactSheet
from marketagents.grounding import check_news_content
from marketagents.interpretation import mark_sentences, prompt_table
from marketagents.llm import message_text
from marketagents.verifier import CITATION, cite, mark_unverified, render, verify

MAX_FIXES = 1
MAX_HISTORY = 6


@dataclass
class Answer:
    text: str
    verified: bool


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return ""
    turns = "\n".join(f"Pytanie: {q}\nOdpowiedź: {a}" for q, a in history[-MAX_HISTORY:])
    return f"\nWcześniejsza rozmowa:\n{turns}\n"


def answer_question(
    llm: BaseChatModel,
    ticker: str,
    question: str,
    report: str,
    facts: FactSheet,
    history: list[tuple[str, str]] | None = None,
    news: list[dict] | None = None,
    company: str = "",
) -> Answer:
    """Odpowiedź na pytanie; history to wcześniejsze pary (pytanie, odpowiedź).

    report to tekst raportu z odwołaniami {{id}} i numerami wiadomości [n] (szkic przed
    podstawieniem wartości), news to ta sama ponumerowana lista wiadomości co w raporcie.
    """
    from marketagents.graph import _format_news

    news = news or []
    prompt = prompts.QA.format(
        ticker=ticker,
        rules=prompts.PLACEHOLDER_RULES,
        facts=prompt_table(facts),
        report=report,
        news=_format_news(news) or "(brak wiadomości)",
        history=_format_history(history or []),
        question=question,
    )
    draft = message_text(llm.invoke(prompt))
    for attempt in range(MAX_FIXES + 1):
        result = verify(draft, facts)
        content = check_news_content(draft, news, (ticker, company))
        bad = sorted({n for n in CITATION.findall(draft) if not 1 <= int(n) <= len(news)})
        issues = result.issues() + [c.message for c in content]
        issues += [f"Numer [{n}] nie odpowiada żadnej wiadomości" for n in bad]
        if not issues:
            return Answer(cite(result.rendered, news), verified=True)
        if attempt == MAX_FIXES:
            break
        fix = prompts.QA_FIX.format(issues="\n".join(f"- {i}" for i in issues), draft=draft)
        draft = message_text(llm.invoke(prompt + "\n\n" + fix))
    flagged = result.flagged_sentences + [c.sentence for c in content]
    marked = render(mark_unverified(mark_sentences(draft, flagged), result.to_mark), facts)
    return Answer(cite(marked, news), verified=False)
