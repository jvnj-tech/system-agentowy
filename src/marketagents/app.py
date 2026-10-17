"""Interfejs w Streamlit: raport, wykresy i asystent pytań.

uv run streamlit run src/marketagents/app.py
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from marketagents import analytics
from marketagents.charts import (
    drawdown_chart,
    fundamentals_chart,
    macro_chart,
    margins_chart,
    monthly_heatmap,
    relative_chart,
    technical_chart,
    volatility_chart,
)
from marketagents.cli import STEP_NAMES, load_market_data
from marketagents.config import load_settings
from marketagents.data import universe
from marketagents.data.cache import DiskCache
from marketagents.data.fundamentals import get_fundamentals
from marketagents.data.macro import get_macro
from marketagents.data.prices import get_prices
from marketagents.evaluation import UsageCounter, append_record, report_record
from marketagents.facts import FactSheet
from marketagents.graph import build_graph, run_report_with_fallback
from marketagents.indicators import MovingAverages, compute_all
from marketagents.llm import AllModelsExhausted, get_chat_model
from marketagents.qa import answer_question

# ETF odwzorowujący NASDAQ-100, notowany na tych samych sesjach co spółki.
BENCHMARK, BENCHMARK_NAME = "QQQ", "NASDAQ-100 (QQQ)"
VOLATILITY_WINDOW = 30
PERIODS = {"3 mies.": 63, "6 mies.": 126, "1 rok": 252, "2 lata": 504, "5 lat": 1260}


@st.cache_resource
def resources():
    settings = load_settings()
    return settings, DiskCache(settings.cache_dir, offline=settings.offline)


@st.cache_data(ttl=3600, show_spinner=False)
def companies_now() -> universe.Universe:
    # Skład NASDAQ-100 z bieżącego dnia (odświeżany raz na dobę, kopia w pliku).
    return universe.current(resources()[1])


@st.cache_resource
def chat_model():
    # Jeden model na cały proces, żeby limit zapytań na minutę był wspólny.
    return get_chat_model(resources()[0])


@st.cache_data(ttl=3600, show_spinner="Pobieranie notowań...")
def chart_data(
    ticker: str, sma: tuple[int, int, int], ema: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    settings, cache = resources()
    prices = get_prices(cache, ticker)
    try:
        annual = get_fundamentals(cache, ticker, settings.sec_user_agent)
    except Exception:  # noqa: BLE001
        annual = None
    return prices, compute_all(prices, MovingAverages(sma, ema)), annual


@st.cache_data(ttl=3600, show_spinner=False)
def benchmark_close() -> pd.Series | None:
    try:
        return get_prices(resources()[1], BENCHMARK)["close"]
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def macro_data() -> pd.DataFrame | None:
    settings, cache = resources()
    try:
        return analytics.macro_history(get_macro(cache, settings.fred_api_key))
    except Exception:  # noqa: BLE001
        return None


def describe_error(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, AllModelsExhausted):
        return text + " Listę modeli ustawia zmienna LLM_MODELS w pliku .env."
    if any(m in text for m in ("RESOURCE_EXHAUSTED", "rate_limit", "429")):
        return (
            "wyczerpano limit zapytań do modelu w darmowym planie. Spróbuj później albo zmień "
            "model lub dostawcę (zmienne LLM_PROVIDER, GEMINI_MODEL, GROQ_MODEL w pliku .env)."
        )
    if "UNAVAILABLE" in text or "503" in text:
        return "model jest chwilowo przeciążony. Spróbuj ponownie za kilka minut."
    return text


# Liczby z karty faktów są podkreślone; podpowiedź pokazuje ich źródło.
# Podpowiedź ze źródłem liczby pokazywana po kliknięciu (fokus) lub najechaniu kursorem.
FACT_STYLE = """<style>
.fact { border-bottom: 1px dotted; cursor: pointer; position: relative; }
.fact:focus { outline: none; background: rgba(128, 128, 128, 0.2); border-radius: 3px; }
.fact:hover::after, .fact:focus::after {
  content: attr(data-zrodlo); position: absolute; left: 0; top: 1.6em; z-index: 10;
  width: max-content; max-width: 24rem; padding: 0.4rem 0.6rem; border-radius: 6px;
  background: #262730; color: #fafafa; font-size: 0.8rem; line-height: 1.35;
  white-space: normal; border: 1px solid rgba(250, 250, 250, 0.25);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}
</style>

"""


def generate_report(ticker: str, ma: MovingAverages) -> dict:
    """Uruchamia graf agentów i pokazuje postęp kolejnych etapów."""
    settings, cache = resources()
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

        def on_restart(spec: str) -> None:
            nonlocal editor_runs
            editor_runs = 0
            st.write(
                f"⚠ {spec} wyczerpał limit zapytań; raport zaczyna się od nowa "
                "na następnym modelu z listy."
            )

        usage = UsageCounter()
        state = run_report_with_fallback(
            chat_model(),
            lambda llm: build_graph(
                llm,
                lambda t: load_market_data(settings, cache, t, ma),
                settings.editor_review,
            ),
            ticker,
            on_step,
            config={"callbacks": [usage]},
            on_restart=on_restart,
        )
        elapsed = time.perf_counter() - start
        status.update(label=f"Raport gotowy po {elapsed:.0f} s", state="complete", expanded=False)
    # Każdy raport trafia do pliku wyników ewaluacji (marketagents-eval podsumowanie).
    try:
        append_record(report_record(ticker, state, elapsed, usage, models_description()))
    except OSError as exc:
        st.caption(f"Nie udało się zapisać wyników ewaluacji: {exc}")
    return {
        **state,
        "elapsed": elapsed,
        "ma": ma,
        "llm_calls": usage.calls,
        "tokens": usage.tokens(),
        "models": dict(usage.models),
    }


def current_model() -> str:
    """Model, który napisze następny raport (pierwszy dostępny z listy)."""
    pool = chat_model()
    available = pool.available()
    if not available:
        return "brak dostępnego modelu (wyczerpane limity)"
    provider, _, name = available[0].partition(":")
    return f"{name} ({'Groq' if provider == 'groq' else 'Gemini'})"


def models_description() -> str:
    """Modele w puli w kolejności użycia, z informacją o wyczerpanych limitach."""
    pool = chat_model()
    blocked = pool.blocked_until()
    names = []
    for spec in pool.specs:
        if spec in blocked:
            until = time.strftime("%H:%M", time.localtime(blocked[spec]))
            names.append(f"{spec} (limit wyczerpany do ok. {until})")
        else:
            names.append(spec)
    return ", ".join(names)


def sidebar() -> tuple[str | None, bool, MovingAverages | None]:
    index = companies_now()
    labels = [f"{c['ticker']} — {c['name']}" for c in index.companies]
    with st.sidebar:
        st.header("Spółka")
        choice = st.selectbox("Spółka z NASDAQ-100", labels, index=0)
        # System jest projektowany i testowany na spółkach z NASDAQ-100, więc wybór
        # ogranicza się do aktualnego składu indeksu.
        ticker = choice.split(" — ")[0]
        cached = ticker in st.session_state.reports
        label = "Wygeneruj raport ponownie" if cached else "Wygeneruj raport"
        # Okna średnich kroczących z pliku .env (SMA_WINDOWS, EMA_WINDOW).
        ma = resources()[0].moving_averages
        clicked = st.button(label, type="primary", width="stretch")
        st.caption(f"Model: {current_model()}")
    return ticker, clicked, ma


def show_report(ticker: str, result: dict) -> None:
    if result["verified"]:
        st.success(
            "Raport zweryfikowany: wszystkie liczby pochodzą z karty faktów, ich opis zgadza się "
            "z ocenami wyznaczonymi przez kod, a zdania o wiadomościach wskazują źródła "
            f"(wersje redakcji: {result['attempts']}, poprawki analityków: "
            f"{result.get('analyst_fixes', 0)}, czas: {result['elapsed']:.0f} s). "
            "Twierdzeń bez liczb weryfikator nie sprawdza."
        )
    else:
        st.warning(
            "Części liczb, ich opisów lub źródeł nie udało się zweryfikować; są oznaczone "
            "w tekście jako [niezweryfikowane], [niezweryfikowana interpretacja] "
            f"lub [brak źródła] (wersje redakcji: {result['attempts']})."
        )
    input_tokens, output_tokens = result.get("tokens", (None, None))
    if input_tokens is not None:
        st.caption(
            f"Wywołania modelu: {result['llm_calls']}, tokeny: {input_tokens} wejściowych "
            f"i {output_tokens} wyjściowych."
            + (
                " Odpowiedzi: " + ", ".join(f"{m} ({n})" for m, n in result["models"].items()) + "."
                if result.get("models")
                else ""
            )
        )
    st.markdown(FACT_STYLE + result.get("report_html", result["report"]), unsafe_allow_html=True)
    st.caption("Kliknij liczbę lub najedź na nią kursorem, aby zobaczyć jej źródło.")
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
                [
                    {
                        "identyfikator": f.id,
                        "opis": f.label,
                        "wartość": f.formatted(),
                        "źródło": f.source,
                    }
                    for f in facts
                ]
            ),
            hide_index=True,
            width="stretch",
        )


def show_charts(ticker: str, ma: MovingAverages) -> None:
    try:
        prices, indicators, annual = chart_data(ticker, ma.sma, ma.ema)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Nie udało się pobrać notowań {ticker}: {exc}")
        return
    period = st.segmented_control("Okres", list(PERIODS), default="1 rok") or "1 rok"
    sessions = PERIODS[period]
    close = prices["close"].iloc[-sessions:]
    st.plotly_chart(
        technical_chart(prices.iloc[-sessions:], indicators.iloc[-sessions:], ma),
        width="stretch",
    )

    st.subheader(f"{ticker} na tle {BENCHMARK_NAME}")
    benchmark = benchmark_close()
    if benchmark is None or ticker == BENCHMARK:
        st.caption("Brak notowań indeksu do porównania.")
    else:
        relative = analytics.relative_performance(close, benchmark)
        st.plotly_chart(relative_chart(relative, ticker, BENCHMARK_NAME), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Obsunięcie od szczytu")
        st.plotly_chart(drawdown_chart(analytics.drawdown(close)), width="stretch")
    with right:
        st.subheader(f"Zmienność ({VOLATILITY_WINDOW} sesji, w skali roku)")
        volatility = analytics.rolling_volatility(prices["close"], VOLATILITY_WINDOW)
        st.plotly_chart(
            volatility_chart(volatility.iloc[-sessions:], VOLATILITY_WINDOW), width="stretch"
        )

    st.subheader("Miesięczne stopy zwrotu (%)")
    st.plotly_chart(monthly_heatmap(analytics.monthly_returns(prices["close"])), width="stretch")
    st.caption("Cała dostępna historia notowań, niezależnie od wybranego okresu.")

    if annual is not None and not annual.empty:
        left, right = st.columns(2)
        with left:
            st.subheader("Przychody i zysk netto (10-K)")
            st.plotly_chart(fundamentals_chart(annual.iloc[-10:]), width="stretch")
        margins = analytics.margins(annual.iloc[-10:])
        with right:
            st.subheader("Marże (10-K)")
            if margins.empty:
                st.caption("Brak danych o przychodach.")
            else:
                st.plotly_chart(margins_chart(margins), width="stretch")
    else:
        st.caption("Brak danych ze sprawozdań finansowych (sprawdź SEC_USER_AGENT).")

    st.subheader("Otoczenie makroekonomiczne w USA")
    macro = macro_data()
    if macro is None or macro.empty:
        st.caption("Brak danych makro (sprawdź klucz FRED_API_KEY).")
    else:
        st.plotly_chart(macro_chart(macro), width="stretch")


def show_chat(ticker: str, result: dict | None) -> None:
    if result is None:
        st.info("Najpierw wygeneruj raport — asystent odpowiada na pytania o jego treść.")
        return
    chat = st.session_state.chats.setdefault(ticker, [])
    for message in chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("verified") is False:
                st.caption(
                    "Części odpowiedzi nie udało się zweryfikować; jest oznaczona w tekście."
                )

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
                result.get("draft") or result["report"],
                FactSheet.from_records(result["facts"]),
                history,
                news=result.get("news", []),
                company=result.get("company", ""),
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

    ticker, clicked, ma = sidebar()
    if ticker is None:
        return
    st.title(ticker)

    if clicked:
        try:
            st.session_state.reports[ticker] = generate_report(ticker, ma)
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
        show_charts(ticker, ma)
    with chat_tab:
        show_chat(ticker, result)


main()
