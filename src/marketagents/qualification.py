"""Test kwalifikacyjny modelu językowego.

Model trafia do puli modeli (LLM_MODELS) tylko wtedy, gdy na tych samych zapisanych danych
napisze kompletne raporty, które przejdą weryfikację. Kryteria są te same dla każdego
modelu, więc przełączenie na inny model po wyczerpaniu limitu nie obniża jakości raportu
poniżej przyjętego progu:

1. każdy analityk, dla którego są dane, i redaktor zwracają niepustą odpowiedź,
2. raport zawiera wszystkie sekcje (z ostatnią sekcją ograniczeń),
3. po poprawkach w raporcie zostaje najwyżej MAX_MARKED oznaczeń (liczby, interpretacje,
   brakujące źródła),
4. raport powstaje w czasie nie dłuższym niż MAX_SECONDS.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

MAX_MARKED = 1
MAX_SECONDS = 600
DEFAULT_TICKERS = ("AAPL", "LULU")
RESULTS_FILE = "kwalifikacja.json"


def evaluate_state(state: dict, expected_analysts: int, elapsed: float) -> dict:
    """Wynik jednego raportu względem kryteriów."""
    from marketagents.graph import LAST_SECTION

    draft = state.get("draft", "")
    marked = (
        len(state.get("unverified", []))
        + len(state.get("flagged", []))
        + len(state.get("uncited", []))
    )
    checks = {
        "odpowiedzi niepuste": bool(draft.strip())
        and len(state.get("analyses", {})) == expected_analysts,
        "raport kompletny": LAST_SECTION in draft,
        f"oznaczenia ≤ {MAX_MARKED}": marked <= MAX_MARKED,
        f"czas ≤ {MAX_SECONDS} s": elapsed <= MAX_SECONDS,
    }
    return {
        "zaliczony": all(checks.values()),
        "kryteria": checks,
        "oznaczenia": marked,
        "wersje redakcji": state.get("attempts"),
        "czas_s": round(elapsed, 1),
    }


def _expected_analysts(data) -> int:
    from marketagents.facts import FUNDAMENTAL, MACRO, TECHNICAL

    groups = {f.group for f in data.facts}
    return sum(g in groups for g in (TECHNICAL, FUNDAMENTAL, MACRO)) + bool(data.news)


def test_data(settings, ticker: str):
    """Dane testu: zapisane przy pierwszym teście i odtąd takie same dla każdego modelu."""
    from marketagents.cli import load_market_data
    from marketagents.data.cache import DiskCache
    from marketagents.facts import FactSheet
    from marketagents.graph import MarketData

    path = Path(settings.results_dir) / "kwalifikacja_dane" / f"{ticker}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return MarketData(facts=list(FactSheet.from_records(saved["facts"])), news=saved["news"])
    cache = DiskCache(settings.cache_dir, offline=settings.offline)
    data = load_market_data(settings, cache, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"facts": FactSheet(data.facts).to_records(), "news": data.news},
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return data


def qualify(spec: str, settings, tickers=DEFAULT_TICKERS) -> dict:
    """Uruchamia raporty dla jednego modelu (bez puli) na zapisanych danych."""
    from marketagents.graph import build_graph, run_report
    from marketagents.llm import AllModelsExhausted, ModelPool, make_model

    # Pula z jednym modelem: limit minutowy jest odczekiwany, a dobowy przerywa test.
    model = ModelPool(specs=[spec], models=[make_model(spec, settings)])
    results = {}
    for ticker in tickers:
        data = test_data(settings, ticker)
        graph = build_graph(model, lambda t, data=data: data)
        start = time.perf_counter()
        try:
            state = run_report(graph, ticker)
        except AllModelsExhausted as exc:
            if "too_large" in exc.kinds:
                # Plan dostępu nie pozwala na odpowiedź o długości raportu: model odpada.
                reason = "odpowiedź przekracza limit planu (np. tokeny wyjściowe na minutę)"
                return {"zaliczony": False, "powód": f"{ticker}: {reason}"}
            # Chwilowy limit to nie wada modelu: test trzeba powtórzyć później.
            return {"zaliczony": None, "przerwany": f"{ticker}: {exc}"}
        results[ticker] = evaluate_state(
            state, _expected_analysts(data), time.perf_counter() - start
        )
    return {
        "zaliczony": all(r["zaliczony"] for r in results.values()),
        "data": datetime.now().isoformat(timespec="seconds"),
        "spółki": results,
    }


def load_results(results_dir: Path) -> dict:
    path = Path(results_dir) / RESULTS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_result(results_dir: Path, spec: str, result: dict) -> None:
    path = Path(results_dir) / RESULTS_FILE
    data = load_results(results_dir)
    data[spec] = result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def qualified(specs: list[str], results_dir: Path) -> list[str]:
    """Modele z zaliczonym testem, w kolejności z konfiguracji.

    Dopóki żaden model z listy nie był testowany, lista nie jest zawężana (system działa
    jak przed wprowadzeniem testu); po pierwszym teście używane są tylko modele zaliczone.
    """
    results = load_results(results_dir)
    tested = [s for s in specs if results.get(s, {}).get("zaliczony") is not None]
    if not tested:
        return list(specs)
    return [s for s in specs if results.get(s, {}).get("zaliczony") is True]
