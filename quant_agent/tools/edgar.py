"""SEC EDGAR fetcher: downloads filings into the doc KB and indexes them.

The PIT guardrail here: filings with a filing date after the as-of date are
skipped, so the agent cannot pull a document the researcher could not have
seen. SEC asks for an identifying User-Agent; set SEC_IDENTITY in .env.
"""
import os
import re
import time
from datetime import date

import requests

from .. import config
from ..schemas import Evidence
from . import doc_kb
from .base import Tool


def _headers() -> dict:
    return {"User-Agent": os.environ.get("SEC_IDENTITY", f"quant-agent-v{config.APP_VERSION} research@example.com")}


def _cik(ticker: str):
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=_headers(), timeout=30)
    r.raise_for_status()
    for row in r.json().values():
        if row["ticker"].upper() == ticker.upper():
            return f"{int(row['cik_str']):010d}", row["title"]
    raise RuntimeError(f"Ticker {ticker} not found on SEC EDGAR.")


class EdgarFetchTool(Tool):
    name = "edgar_fetch"
    needs_query = True  # receives the user's query for the ticker-allowlist gate
    description = (
        "edgar_fetch(ticker: str, form: str = '10-K', count: int = 1) - download the most recent SEC "
        "filings of that form filed on or before the as-of date, save them into the doc knowledge base, "
        "and index them so doc_search can query their contents. Only tickers explicitly named in the "
        "user's question may be fetched."
    )

    @staticmethod
    def _query_tickers(query: str) -> set[str]:
        from .base import query_tickers
        return query_tickers(query)

    def run(self, as_of, ticker, form="10-K", count=1, _query="", **_):
        from .base import assert_ticker_named
        requested = str(ticker).upper().strip()
        # GUARDRAIL (Finding 1): refuse to fetch a ticker the user never named.
        assert_ticker_named(requested, _query, self.name)
        cik, title = _cik(requested)
        subs = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=_headers(), timeout=30)
        subs.raise_for_status()
        recent = subs.json()["filings"]["recent"]
        saved = []
        rows = zip(recent["form"], recent["filingDate"], recent["accessionNumber"], recent["primaryDocument"])
        for form_i, fdate, acc, docname in rows:
            if form_i != str(form):
                continue
            fd = date.fromisoformat(fdate)
            if fd > as_of:  # point-in-time guardrail
                continue
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{docname}"
            html = requests.get(url, headers=_headers(), timeout=60).text
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))
            fname = f"{fdate}_{str(ticker).upper()}_{str(form).replace('/', '-')}.txt"
            (config.DOCS_DIR / fname).write_text(text, encoding="utf-8")
            saved.append((fname, fd))
            time.sleep(0.2)  # be polite to SEC
            if len(saved) >= int(count):
                break
        if not saved:
            raise RuntimeError(f"No {form} filings for {ticker} filed on or before {as_of}.")
        doc_kb.ingest()
        listing = ", ".join(f"{f} (filed {d})" for f, d in saved)
        return [
            Evidence(
                source_id=f"edgar:{str(ticker).upper()}:{form}:{as_of}",
                store="edgar",
                content=(
                    f"Downloaded and indexed {len(saved)} {form} filing(s) for {title}: {listing}. "
                    "Use doc_search to query their contents."
                ),
                pub_date=saved[0][1],
                pit_safe=True,
            )
        ]
