"""Interfejs w Streamlit: raport, wykresy i asystent pytań.

uv run streamlit run src/marketagents/app.py
"""

from __future__ import annotations

import re
import time

import pandas as pd
import streamlit as st

from marketagents.charts import fundamentals_chart, technical_chart
from marketagents.cli import STEP_NAMES, load_market_data
from marketagents.config import load_settings
from marketagents.data.cache import DiskCache
from marketagents.data.fundamentals import get_fundamentals
from marketagents.data.prices import get_prices
from marketagents.data.universe import nasdaq100
from marketagents.facts import FactSheet
from marketagents.graph import build_graph, run_report
from marketagents.indicators import compute_all
from marketagents.llm import get_chat_model
from marketagents.qa import answer_question

TICKER = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
PERIODS = {"3 mies.": 63, "6 mies.": 126, "1 rok": 252, "2 lata": 504, "5 lat": 1260}


@st.cache_resource
def resources():
    settings = load_settings()
    return settings, DiskCache(settings.cache_dir, offline=settings.offline)


@st.cache_resource
def chat_model():
    # Jeden model na cały proces, żeby limit zapytań na minutę był wspólny.
    return get_chat_model(resources()[0])


@st.cache_data(ttl=3600, show_spinner="Pobieranie notowań...")
def chart_data(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    settings, cache = resources()
    prices = get_prices(cache, ticker)
    try:
        annual = get_fundamentals(cache, ticker, settings.sec_user_agent)
    except Exception:  # noqa: BLE001
        annual = None
    return prices, compute_all(prices), annual


def describe_error(exc: Exception) -> str:
    if "RESOURCE_EXHAUSTED" in str(exc):
        return (
            "wyczerpano limit zapytań do Gemini w darmowym planie. Spróbuj później "
            "albo ustaw inny model w zmiennej GEMINI_MODEL (limity są osobne dla modeli)."
        )
    if "UNAVAILABLE" in str(exc):
        return "model Gemini jest chwilowo przeciążony. Spróbuj ponownie za kilka minut."
    return str(exc)


def generate_report(ticker: str) -> dict:
    """Uruchamia graf agentów i pokazuje postęp kolejnych etapów."""
    settings, cache = resources()
    graph = build_graph(chat_model(), lambda t: load_market_data(settings, cache, t))
    start = time.perf_counter()
    editor_runs = 0

    with st.status(f"Generowanie raportu dla {ticker}...", expanded=True) as status:

        def on_step(node: str) -> None:
            nonlocal editor_runs
            name = STEP_NAMES.get(node, node)
            if node == "editor":
                editor_runs += 1
                if editor_runs > 1:
                    name += f" (poprawka {editor_runs - 1})"
            st.write(f"✓ {name} — {time.perf_counter() - start:.0f} s")

        state = run_report(graph, ticker, on_step)
        elapsed = time.perf_counter() - start
        status.update(label=f"Raport gotowy po {elapsed:.0f} s", state="complete", expanded=False)
    return {**state, "elapsed": elapsed}


def sidebar() -> tuple[str | None, bool]:
    companies = nasdaq100()
    labels = [f"{c['ticker']} — {c['name']}" for c in companies]
    with st.sidebar:
        st.header("Spółka")
        choice = st.selectbox("Spółka z NASDAQ-100", labels, index=0)
        typed = st.text_input("lub symbol spoza listy", placeholder="np. INTC").strip().upper()
        ticker = typed or choice.split(" — ")[0]
        if not TICKER.match(ticker):
            st.error("Niepoprawny symbol spółki.")
            return None, False
        cached = ticker in st.session_state.reports
        label = "Wygeneruj raport ponownie" if cached else "Wygeneruj raport"
        clicked = st.button(label, type="primary", width="stretch")
        st.caption("Raport tworzą agenci na modelu Gemini; generowanie trwa zwykle kilka minut.")
    return ticker, clicked


def show_report(ticker: str, result: dict) -> None:
    if result["verified"]:
        st.success(
            "Raport zweryfikowany: wszystkie liczby pochodzą z karty faktów "
            f"(wersje redakcji: {result['attempts']}, czas: {result['elapsed']:.0f} s)."
        )
    else:
        st.warning(
            "Części liczb nie udało się zweryfikować; są oznaczone w tekście jako "
            f"[niezweryfikowane] (wersje redakcji: {result['attempts']})."
        )
    st.markdown(result["report"])
    st.download_button(
        "Pobierz raport (.md)",
        result["report"],
        file_name=f"raport_{ticker}.md",
        mime="text/markdown",
    )
    with st.expander("Karta faktów"):
        facts = FactSheet.from_records(result["facts"])
        st.dataframe(
            pd.DataFrame(
                [{"identyfikator": f.id, "opis": f.label, "wartość": f.formatted()} for f in facts]
            ),
            hide_index=True,
            width="stretch",
        )


def show_charts(ticker: str) -> None:
    try:
        prices, indicators, annual = chart_data(ticker)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Nie udało się pobrać notowań {ticker}: {exc}")
        return
    period = st.segmented_control("Okres", list(PERIODS), default="1 rok") or "1 rok"
    sessions = PERIODS[period]
    st.plotly_chart(
        technical_chart(prices.iloc[-sessions:], indicators.iloc[-sessions:]),
        width="stretch",
    )
    if annual is not None and not annual.empty:
        st.subheader("Przychody i zysk netto (sprawozdania 10-K)")
        st.plotly_chart(fundamentals_chart(annual.iloc[-10:]), width="stretch")


def show_chat(ticker: str, result: dict | None) -> None:
    if result is None:
        st.info("Najpierw wygeneruj raport — asystent odpowiada na pytania o jego treść.")
        return
    chat = st.session_state.chats.setdefault(ticker, [])
    for message in chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("verified") is False:
                st.caption("Odpowiedź zawiera niezweryfikowane liczby.")

    question = st.chat_input(f"Zadaj pytanie o raport {ticker}")
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    history = [(q["content"], a["content"]) for q, a in zip(chat[::2], chat[1::2], strict=False)]
    with st.chat_message("assistant"), st.spinner("Asystent przygotowuje odpowiedź..."):
        try:
            answer = answer_question(
                chat_model(),
                ticker,
                question,
                result["report"],
                FactSheet.from_records(result["facts"]),
                history,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Nie udało się uzyskać odpowiedzi: {describe_error(exc)}")
            return
    chat += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer.text, "verified": answer.verified},
    ]
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Analiza spółek", layout="wide")
    # Raporty i rozmowy są trzymane w sesji, żeby nie generować ich przy każdej interakcji.
    st.session_state.setdefault("reports", {})
    st.session_state.setdefault("chats", {})

    ticker, clicked = sidebar()
    if ticker is None:
        return
    st.title(ticker)

    if clicked:
        try:
            st.session_state.reports[ticker] = generate_report(ticker)
            st.session_state.chats.pop(ticker, None)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Nie udało się wygenerować raportu: {describe_error(exc)}")
    result = st.session_state.reports.get(ticker)

    # Klucz zachowuje wybraną zakładkę po odświeżeniu strony, np. po odpowiedzi asystenta.
    report_tab, charts_tab, chat_tab = st.tabs(
        ["Raport", "Wykresy", "Pytania"], key="tab", on_change="rerun"
    )
    with report_tab:
        if result:
            show_report(ticker, result)
        else:
            st.info("Wybierz spółkę i kliknij „Wygeneruj raport”.")
    with charts_tab:
        show_charts(ticker)
    with chat_tab:
        show_chat(ticker, result)


main()
