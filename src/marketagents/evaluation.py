"""Ewaluacja jakości raportów.

uv run marketagents-eval raporty AAPL MSFT NVDA          raporty z kartą faktów
uv run marketagents-eval raporty AAPL MSFT --bez-karty   wariant porównawczy bez karty
uv run marketagents-eval podsumowanie                    zestawienie zapisanych raportów
uv run marketagents-eval bledy [--sedzia]                test wstrzykniętych błędów

Pomiary:
1. Raport z kartą faktów: czas, liczba wywołań modelu i tokenów (o ile dostawca je podaje),
   poprawki analityków, liczba wersji redakcji, problemy znalezione w kolejnych wersjach,
   oznaczone liczby i zdania. Każdy raport wygenerowany w aplikacji albo poleceniem „raporty” jest
   dopisywany do pliku wyników w formacie JSON Lines.
2. Wariant bez karty faktów: analitycy i redaktor dostają te same dane jako zwykłą tabelę
   i sami przepisują liczby, bez odwołań i bez weryfikatora. Liczone są liczby, których
   nie ma w karcie faktów. Bez odwołań nie da się przypisać słów do faktów, więc w tym
   wariancie opis liczb nie jest sprawdzany.
3. Wstrzyknięte błędy: w raportach zweryfikowanych podmieniana jest po jednej liczbie,
   słownym opisie, okresie zmiany albo roku. Dla każdej zmiany sprawdzane jest, czy wykryje
   ją weryfikator, a opcjonalnie także model językowy w roli sędziego. Sędzia ocenia też
   raporty bez zmian, żeby policzyć jego fałszywe alarmy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

from marketagents import prompts
from marketagents.coverage import NOTE_PREFIX
from marketagents.facts import FUNDAMENTAL, MACRO, TECHNICAL, Fact, FactSheet, format_number
from marketagents.graph import MarketData, _format_news
from marketagents.interpretation import (
    AVERAGE,
    DIRECTIONAL,
    DISTANCE,
    HEDGE,
    NEGATION,
    PERIODS,
    SENTENCE,
    label,
    periods_in,
)
from marketagents.llm import message_text
from marketagents.verifier import (
    DEFAULT_ALLOWED,
    INDEX_NAME,
    ISO_DATE,
    LIST_MARKER,
    NUMBER,
    PLACEHOLDER,
    YEAR,
    fact_years,
    indicator_periods,
    label_numbers,
    matches_fact,
    parse_number,
    render,
    verify,
)

DEFAULT_RESULTS = Path("wyniki") / "raporty.jsonl"
WITH_CARD, WITHOUT_CARD = "karta", "bez-karty"
# Data w zapisie 02.10.2026, którą model może wpisać w wariancie bez karty.
DMY_DATE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b")

ANALYSTS = [
    ("techniczna", prompts.TECHNICAL_ANALYST, TECHNICAL, False),
    ("fundamentalna", prompts.FUNDAMENTAL_ANALYST, FUNDAMENTAL, False),
    ("wiadomości", prompts.NEWS_ANALYST, "", True),
    ("makroekonomiczna", prompts.MACRO_ANALYST, MACRO, False),
]


class UsageCounter(BaseCallbackHandler):
    """Liczy wywołania modelu i tokeny; analitycy działają równolegle, stąd blokada."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reported = False
        # Liczba odpowiedzi każdego modelu z puli (po przełączeniu raport pisze kilka modeli).
        self.models: dict[str, int] = {}

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        messages = [
            g.message
            for gens in response.generations
            for g in gens
            if isinstance(getattr(g, "message", None), AIMessage)
        ]
        usage = [m.usage_metadata for m in messages if m.usage_metadata]
        with self._lock:
            self.calls += 1
            for m in messages:
                spec = m.response_metadata.get("model_spec")
                if spec:
                    self.models[spec] = self.models.get(spec, 0) + 1
            for u in usage:
                self.reported = True
                self.input_tokens += u.get("input_tokens", 0)
                self.output_tokens += u.get("output_tokens", 0)

    def tokens(self) -> tuple[int | None, int | None]:
        if not self.reported:
            return None, None
        return self.input_tokens, self.output_tokens


MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")


def unsupported_numbers(text: str, facts: FactSheet) -> list[str]:
    """Liczby w gotowym tekście, których nie ma w karcie faktów (porównanie z dowolnym faktem).

    Kryterium jest łagodniejsze niż w weryfikatorze, który nie przepuszcza żadnej liczby
    wpisanej poza odwołaniem, bo w wariancie bez karty zdania nie mają odwołań.
    """
    allowed = DEFAULT_ALLOWED | indicator_periods(facts) | label_numbers(facts)
    years = fact_years(facts)
    # Opis zakresu danych dopisuje kod, a odnośniki do wiadomości zawierają daty i adresy.
    text = MARKDOWN_LINK.sub(" ", text.split(NOTE_PREFIX)[0])
    stripped = DMY_DATE.sub(" ", ISO_DATE.sub(" ", PLACEHOLDER.sub(" ", text)))
    stripped = LIST_MARKER.sub(" ", INDEX_NAME.sub(" ", stripped))
    out = [m.group(0) for m in YEAR.finditer(stripped) if int(m.group(0)) not in years]
    for match in NUMBER.finditer(YEAR.sub(" ", stripped)):
        raw = match.group(0)
        if raw.isdigit() and int(raw) in allowed:
            continue
        if not any(matches_fact(parse_number(raw), fact) for fact in facts):
            out.append(raw)
    return out


def report_record(
    ticker: str,
    state: dict,
    elapsed: float,
    usage: UsageCounter,
    model: str = "",
    variant: str = WITH_CARD,
) -> dict:
    """Wiersz pliku wyników dla jednego raportu."""
    facts = FactSheet.from_records(state["facts"])
    input_tokens, output_tokens = usage.tokens()
    return {
        "ticker": ticker,
        "variant": variant,
        "model": ", ".join(usage.models) or model,
        "models": dict(usage.models),
        "created": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 1),
        "llm_calls": usage.calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "analyst_fixes": state.get("analyst_fixes", 0),
        "attempts": state.get("attempts"),
        "restarts": state.get("restarts", 0),
        "review_issues": state.get("review_issues", []),
        "verified": state.get("verified"),
        "issue_counts": state.get("issue_counts", []),
        "unverified": len(state.get("unverified", [])),
        "flagged": len(state.get("flagged", [])),
        "unsupported_numbers": unsupported_numbers(state.get("report", ""), facts),
        "draft": state.get("draft", ""),
        "report": state.get("report", ""),
        "facts": state["facts"],
    }


def append_record(record: dict, path: Path = DEFAULT_RESULTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: Path = DEFAULT_RESULTS) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plain_table(facts: FactSheet) -> str:
    """Tabela danych bez identyfikatorów, do wariantu bez karty faktów."""
    lines = ["| opis | wartość |", "| --- | --- |"]
    lines += [f"| {f.label} | {f.formatted()} |" for f in facts]
    return "\n".join(lines)


def report_without_fact_card(
    llm: BaseChatModel, ticker: str, data: MarketData, config: dict | None = None
) -> dict:
    """Ci sami analitycy i redaktor, ale liczby przepisuje model i nikt ich nie sprawdza."""
    facts = FactSheet(data.facts)
    analyses = {}
    for name, template, group, needs_news in ANALYSTS:
        subset = facts.group(group)
        if needs_news and not data.news:
            continue
        if not needs_news and not any(f.group == group for f in subset):
            continue
        prompt = template.format(
            ticker=ticker,
            rules=prompts.PLAIN_NUMBER_RULES,
            facts=plain_table(subset),
            news=_format_news(data.news),
        )
        analyses[name] = message_text(llm.invoke(prompt, config=config))
    prompt = prompts.EDITOR.format(
        ticker=ticker,
        rules=prompts.PLAIN_NUMBER_RULES,
        facts=plain_table(facts),
        analyses="\n\n".join(f"### Analiza: {k}\n{v}" for k, v in analyses.items()),
    )
    report = message_text(llm.invoke(prompt, config=config))
    return {"ticker": ticker, "facts": facts.to_records(), "draft": report, "report": report}


# --- Wstrzyknięte błędy ---

# Słowo -> słowo przeciwne. Oba słowa należą do jednej grupy: kierunek zmiany, położenie
# kursu względem średniej albo strefa RSI.
FLIP_GROUPS = {
    "kierunek": {
        "wzrósł": "spadł",
        "wzrosła": "spadła",
        "wzrosło": "spadło",
        "wzrosły": "spadły",
        "wzrost": "spadek",
        "wzrostu": "spadku",
        "wzrostem": "spadkiem",
        "zyskał": "stracił",
        "zyskała": "straciła",
        "zyskały": "straciły",
    },
    "średnia": {"powyżej": "poniżej"},
    "rsi": {
        "wykupienia": "wyprzedania",
        "wykupienie": "wyprzedanie",
        "wykupione": "wyprzedane",
        "neutralnej": "wykupienia",
        "neutralną": "wykupienia",
        "neutralna": "wykupienia",
    },
}
FLIP: dict[str, tuple[str, str]] = {}
for _group, _words in FLIP_GROUPS.items():
    for _word, _opposite in _words.items():
        FLIP[_word] = (_opposite, _group)
        FLIP.setdefault(_opposite, (_word, _group))
# Ocena faktu (funkcja label), z którą zgadza się dane słowo w zdaniu bez zmian.
AGREES = {
    "wzrost": {"wzrósł", "wzrosła", "wzrosło", "wzrosły", "wzrost", "wzrostu", "wzrostem"}
    | {"zyskał", "zyskała", "zyskały"},
    "spadek": {"spadł", "spadła", "spadło", "spadły", "spadek", "spadku", "spadkiem"}
    | {"stracił", "straciła", "straciły"},
    "kurs powyżej średniej": {"powyżej"},
    "kurs poniżej średniej": {"poniżej"},
    "wykupienie": {"wykupienia", "wykupienie", "wykupione"},
    "wyprzedanie": {"wyprzedania", "wyprzedanie", "wyprzedane"},
    "strefa neutralna": {"neutralnej", "neutralną", "neutralna"},
}
CHANGES = ("change_1m", "change_3m", "change_1y")
KINDS = ("liczba", "inny fakt", "kierunek", "okres", "rok")


@dataclass(frozen=True)
class Mutation:
    kind: str
    before: str
    after: str
    text: str


def _numeric_refs(draft: str, facts: FactSheet) -> list[re.Match]:
    return [
        m
        for m in PLACEHOLDER.finditer(draft)
        if m.group(1) in facts and not isinstance(facts[m.group(1)].value, str)
    ]


def _sentences(text: str) -> list[tuple[int, str]]:
    """Zdania (podział jak w module interpretation) z pozycją początku w tekście."""
    out, start = [], 0
    for brk in SENTENCE.finditer(text):
        out.append((start, text[start : brk.start()]))
        start = brk.end()
    out.append((start, text[start:]))
    return [(i, sentence) for i, sentence in out if sentence.strip()]


def _value_text(fact: Fact, value: float, draft: str, end: int) -> str:
    """Wartość w zapisie karty; bez jednostki, jeśli autor dopisał ją po odwołaniu."""
    text = Fact(fact.id, fact.label, value, fact.unit, fact.decimals).formatted()
    if fact.unit and re.match(rf"\s*{re.escape(fact.unit)}(?![\w])", draft[end:]):
        text = format_number(value, fact.decimals)
    return text


def _distorted(fact: Fact, rng: random.Random) -> float:
    shift = max(abs(fact.value) * rng.uniform(0.05, 0.3), 5 * 10 ** (-fact.decimals))
    return fact.value + rng.choice([-1, 1]) * shift


def _flip(word: str) -> str:
    flipped = FLIP[word.lower()][0]
    return flipped.capitalize() if word[0].isupper() else flipped


def _mutation(kind: str, draft: str, start: int, end: int, new: str) -> Mutation:
    return Mutation(kind, draft[start:end], new, draft[:start] + new + draft[end:])


def _described_fact(group: str, sentence: str, refs: list[str]) -> str | None:
    """Fakt, którego dotyczy słowo z danej grupy; None, gdy zdanie nie jest jednoznaczne."""
    if group == "kierunek":
        ids = [r for r in refs if DIRECTIONAL.match(r)]
    elif group == "średnia":
        ids = [r for r in refs if AVERAGE.match(r) or DISTANCE.match(r)]
        if "kurs" not in sentence.lower():
            return None
    else:
        ids = [r for r in refs if r == "rsi_14"]
        if HEDGE.search(sentence):
            return None
    return ids[0] if len(set(ids)) == 1 else None


def inject_errors(draft: str, facts: FactSheet, rng: random.Random) -> list[Mutation]:
    """Po jednej zmianie każdego rodzaju, o ile raport zawiera odpowiednie miejsce.

    Słowo i okres są zmieniane tylko w zdaniach, które jednoznacznie dotyczą jednego faktu
    i stwierdzają ocenę zgodną z kartą (bez przeczenia), żeby każda zmiana była rzeczywistą
    niezgodnością z danymi. Z tego powodu miejsca zmian pokrywają się z tymi, które sprawdza
    weryfikator, a jego wykrywalność wynika z konstrukcji testu; porównaniem jest sędzia.
    """
    out: list[Mutation] = []
    refs = _numeric_refs(draft, facts)
    if refs:
        m = rng.choice(refs)
        fact = facts[m.group(1)]
        wrong = _value_text(fact, _distorted(fact, rng), draft, m.end())
        out.append(_mutation("liczba", draft, *m.span(), wrong))

        pairs = [
            (m, other)
            for m in refs
            for other in facts
            if not isinstance(other.value, str)
            and other.id != m.group(1)
            and other.unit == facts[m.group(1)].unit
            and other.formatted() != facts[m.group(1)].formatted()
        ]
        if pairs:
            m, other = rng.choice(pairs)
            wrong = _value_text(other, other.value, draft, m.end())
            out.append(_mutation("inny fakt", draft, *m.span(), wrong))

    words, periods = [], []
    for start, sentence in _sentences(draft):
        if NEGATION.search(sentence):
            continue
        ids = [r for r in PLACEHOLDER.findall(sentence) if r in facts]
        for w in re.finditer(r"\w+", sentence):
            word = w.group(0)
            if word.lower() not in FLIP:
                continue
            fact_id = _described_fact(FLIP[word.lower()][1], sentence, ids)
            # Ocena odległości od średniej zawiera też wielkość („kurs powyżej średniej, …”).
            assessment = (label(fact_id, facts) or "").split(",")[0] if fact_id else ""
            if fact_id and word.lower() in AGREES.get(assessment, set()):
                words.append((start + w.start(), start + w.end(), _flip(word)))
        found = periods_in(PLACEHOLDER.sub(" ", sentence))
        changes = [r for r in ids if r in PERIODS]
        if len(found) != 1 or len(set(changes)) != 1 or changes[0] not in CHANGES:
            continue
        if PERIODS[changes[0]] != found[0]:
            continue
        m = next(m for m in PLACEHOLDER.finditer(sentence) if m.group(1) == changes[0])
        for other in CHANGES:
            if other in facts and PERIODS[other] != found[0]:
                periods.append((start + m.start(), start + m.end(), f"{{{{{other}}}}}"))
    if words:
        out.append(_mutation("kierunek", draft, *rng.choice(words)))
    if periods:
        out.append(_mutation("okres", draft, *rng.choice(periods)))

    dates = [
        m
        for m in PLACEHOLDER.finditer(draft)
        if m.group(1) in facts
        and isinstance(facts[m.group(1)].value, str)
        and ISO_DATE.fullmatch(facts[m.group(1)].value)
    ]
    if dates:
        m = rng.choice(dates)
        year = str(int(facts[m.group(1)].value[:4]) - 3)
        out.append(_mutation("rok", draft, *m.span(), year))
    return out


def judge_table(facts: FactSheet) -> str:
    lines = ["| opis | wartość | ocena |", "| --- | --- | --- |"]
    lines += [f"| {f.label} | {f.formatted()} | {label(f.id, facts) or ''} |" for f in facts]
    return "\n".join(lines)


def judge_finds_error(llm: BaseChatModel, ticker: str, report: str, facts: FactSheet) -> bool:
    """Ocena modelu w roli sędziego: True, gdy uzna, że raport zawiera niezgodność."""
    prompt = prompts.JUDGE.format(ticker=ticker, facts=judge_table(facts), report=report)
    answer = message_text(llm.invoke(prompt)).strip()
    first = answer.splitlines()[0].upper() if answer else ""
    return "BŁĄD" in first


def injection_test(
    records: list[dict],
    seed: int = 0,
    llm: BaseChatModel | None = None,
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Wiersz na każdą zmianę oraz na każdy raport bez zmian (rodzaj „bez błędu”)."""
    rng = random.Random(seed)
    rows = []
    for record in records:
        if record.get("variant") != WITH_CARD:
            continue
        facts = FactSheet.from_records(record["facts"])
        draft = record["draft"]
        if not verify(draft, facts).ok:
            continue
        cases = [Mutation("bez błędu", "", "", draft), *inject_errors(draft, facts, rng)]
        for case in cases:
            row = {
                "ticker": record["ticker"],
                "kind": case.kind,
                "before": case.before,
                "after": case.after,
                "verifier": not verify(case.text, facts).ok,
            }
            if llm is not None:
                row["judge"] = judge_finds_error(
                    llm, record["ticker"], render(case.text, facts), facts
                )
            rows.append(row)
            if on_row:
                on_row(row)
    return rows


# --- Zestawienia ---


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return round(mean(values), 2) if values else None


def summarize_reports(records: list[dict]) -> list[dict]:
    """Jeden wiersz na wariant (z kartą faktów i bez niej)."""
    rows = []
    for variant in (WITH_CARD, WITHOUT_CARD):
        group = [r for r in records if r.get("variant") == variant]
        if not group:
            continue
        row = {
            "wariant": variant,
            "raporty": len(group),
            "śr. czas [s]": _mean([r.get("elapsed_s") for r in group]),
            "śr. wywołania modelu": _mean([r.get("llm_calls") for r in group]),
            "śr. tokeny wejściowe": _mean([r.get("input_tokens") for r in group]),
            "śr. tokeny wyjściowe": _mean([r.get("output_tokens") for r in group]),
            "śr. liczby spoza karty": _mean([len(r["unsupported_numbers"]) for r in group]),
            "raporty z liczbą spoza karty [%]": round(
                100 * sum(bool(r["unsupported_numbers"]) for r in group) / len(group), 1
            ),
        }
        if variant == WITH_CARD:
            row |= {
                "zweryfikowane za 1. razem [%]": round(
                    100 * sum(r["verified"] and r["attempts"] == 1 for r in group) / len(group), 1
                ),
                "zweryfikowane ostatecznie [%]": round(
                    100 * sum(bool(r["verified"]) for r in group) / len(group), 1
                ),
                "śr. poprawki analityków": _mean([r.get("analyst_fixes") for r in group]),
                "śr. wersje redakcji": _mean([r.get("attempts") for r in group]),
                "śr. problemy w 1. wersji": _mean(
                    [r["issue_counts"][0] for r in group if r.get("issue_counts")]
                ),
                "śr. oznaczone liczby": _mean([r.get("unverified") for r in group]),
                "śr. oznaczone zdania": _mean([r.get("flagged") for r in group]),
            }
        rows.append(row)
    return rows


def summarize_injections(rows: list[dict]) -> list[dict]:
    """Odsetek wykrytych zmian według rodzaju; dla „bez błędu” to odsetek fałszywych alarmów."""
    out = []
    for kind in ("bez błędu", *KINDS):
        group = [r for r in rows if r["kind"] == kind]
        if not group:
            continue
        row = {
            "rodzaj": kind,
            "przypadki": len(group),
            "weryfikator [%]": round(100 * sum(r["verifier"] for r in group) / len(group), 1),
        }
        judged = [r for r in group if "judge" in r]
        if judged:
            row["sędzia [%]"] = round(100 * sum(r["judge"] for r in judged) / len(judged), 1)
        out.append(row)
    return out


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "(brak danych)"
    columns = list(dict.fromkeys(k for row in rows for k in row))
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


# --- Polecenie marketagents-eval ---


def _generate(args, settings, cache) -> None:
    from marketagents.cli import load_market_data
    from marketagents.graph import build_graph, run_report_with_fallback
    from marketagents.llm import get_chat_model

    llm = get_chat_model(settings)
    model = ", ".join(getattr(llm, "specs", []))
    variant = WITHOUT_CARD if args.bez_karty else WITH_CARD

    def make_graph(model):
        return build_graph(
            model, lambda t: load_market_data(settings, cache, t), settings.editor_review
        )

    for ticker in [t.upper() for t in args.tickers]:
        usage = UsageCounter()
        config = {"callbacks": [usage]}
        start = time.perf_counter()
        try:
            if variant == WITH_CARD:
                state = run_report_with_fallback(llm, make_graph, ticker, config=config)
            else:
                data = load_market_data(settings, cache, ticker)
                # Wariant porównawczy też pisze jeden model: pierwszy dostępny z puli.
                single = llm.only(llm.available()[0]) if hasattr(llm, "only") else llm
                state = report_without_fact_card(single, ticker, data, config)
        except Exception as exc:  # noqa: BLE001
            print(f"{ticker}: błąd ({exc})", flush=True)
            continue
        record = report_record(ticker, state, time.perf_counter() - start, usage, model, variant)
        append_record(record, Path(args.wyniki))
        status = "" if variant == WITHOUT_CARD else f", zweryfikowany: {record['verified']}"
        print(
            f"{ticker}: {record['elapsed_s']} s, wywołania modelu: {record['llm_calls']}, "
            f"liczby spoza karty: {len(record['unsupported_numbers'])}{status}",
            flush=True,
        )


def _qualify(args, settings) -> None:
    from marketagents.llm import configured_specs
    from marketagents.qualification import DEFAULT_TICKERS, load_results, qualify, save_result

    tickers = [t.upper() for t in (args.spolki or DEFAULT_TICKERS)]
    for spec in args.modele or configured_specs(settings):
        print(f"{spec}: test na {', '.join(tickers)}...", flush=True)
        try:
            previous = load_results(settings.results_dir).get(spec, {})
            done = previous.get("spółki") if previous.get("zaliczony") is None else None
            result = qualify(spec, settings, tickers, done)
        except Exception as exc:  # noqa: BLE001
            # Błąd jednego modelu (np. awaria serwera) nie przerywa testu pozostałych.
            result = {"zaliczony": None, "przerwany": f"błąd: {str(exc)[:200]}"}
        save_result(settings.results_dir, spec, result)
        if result["zaliczony"] is None:
            print(f"  przerwany ({result['przerwany']}); powtórz test później", flush=True)
            continue
        if "powód" in result:
            print(f"  NIEZALICZONY: {result['powód']}", flush=True)
            continue
        print(f"  {'ZALICZONY' if result['zaliczony'] else 'NIEZALICZONY'}", flush=True)
        for ticker, r in result["spółki"].items():
            failed = [k for k, ok in r["kryteria"].items() if not ok]
            print(
                f"  {ticker}: {r['czas_s']} s, oznaczenia {r['oznaczenia']}"
                + (f", niespełnione: {', '.join(failed)}" if failed else ""),
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    from marketagents.config import load_settings
    from marketagents.data.cache import DiskCache

    parser = argparse.ArgumentParser(description="Ewaluacja jakości raportów.")
    parser.add_argument("--wyniki", default=str(DEFAULT_RESULTS), help="plik wyników (.jsonl)")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("raporty", help="wygenerowanie raportów i zapis wyników")
    gen.add_argument("tickers", nargs="+")
    gen.add_argument("--bez-karty", action="store_true", help="wariant bez karty faktów")
    sub.add_parser("podsumowanie", help="zestawienie zapisanych raportów")
    inj = sub.add_parser("bledy", help="test wstrzykniętych błędów")
    inj.add_argument("--sedzia", action="store_true", help="także ocena modelu w roli sędziego")
    inj.add_argument("--ziarno", type=int, default=0, help="ziarno losowania zmian")
    inj.add_argument("--csv", help="zapis wszystkich przypadków do pliku CSV")
    qual = sub.add_parser("kwalifikacja", help="test kwalifikacyjny modeli do puli")
    qual.add_argument(
        "modele", nargs="*", help="np. groq:openai/gpt-oss-120b (domyślnie LLM_MODELS)"
    )
    qual.add_argument("--spolki", nargs="+", help="spółki testu (domyślnie AAPL LULU INTC SBUX)")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.command == "kwalifikacja":
        _qualify(args, settings)
        return
    if args.command == "raporty":
        _generate(args, settings, DiskCache(settings.cache_dir, offline=settings.offline))
        return

    records = load_records(Path(args.wyniki))
    if args.command == "podsumowanie":
        print(markdown_table(summarize_reports(records)))
        return

    llm = None
    if args.sedzia:
        from marketagents.llm import get_chat_model

        llm = get_chat_model(settings)
    rows = injection_test(records, args.ziarno, llm)
    print(markdown_table(summarize_injections(rows)))
    if args.csv:
        import csv

        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
