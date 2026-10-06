"""Asystent odpowiadający na pytania o wygenerowany raport.

Obowiązują go te same zasady co autorów raportu: liczby wstawia jako odwołania do karty
faktów, a odpowiedź przechodzi przez weryfikator. Jeśli zawiera liczbę spoza karty,
model dostaje jedną szansę na poprawkę, a potem niezweryfikowane liczby są oznaczane.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from marketagents import prompts
from marketagents.facts import FactSheet
from marketagents.interpretation import mark_sentences
from marketagents.llm import message_text
from marketagents.verifier import mark_unverified, render, verify

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
) -> Answer:
    """Odpowiedź na pytanie; history to wcześniejsze pary (pytanie, odpowiedź)."""
    prompt = prompts.QA.format(
        ticker=ticker,
        rules=prompts.PLACEHOLDER_RULES,
        facts=facts.as_prompt_table(),
        report=report,
        history=_format_history(history or []),
        question=question,
    )
    draft = message_text(llm.invoke(prompt))
    for attempt in range(MAX_FIXES + 1):
        result = verify(draft, facts)
        if result.ok:
            return Answer(result.rendered, verified=True)
        if attempt == MAX_FIXES:
            break
        issues = "\n".join(f"- {i}" for i in result.issues())
        fix = prompts.QA_FIX.format(issues=issues, draft=draft)
        draft = message_text(llm.invoke(prompt + "\n\n" + fix))
    marked = render(mark_sentences(draft, result.flagged_sentences), facts)
    return Answer(mark_unverified(marked, result.to_mark), verified=False)
