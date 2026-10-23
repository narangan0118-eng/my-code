"""Audit the 2012 S&P 500 historical universe for future-data eligibility bias.

Inputs (same directory):
    candidate_tickers.csv (Ticker column, created by build_historical_candidates.py)
Outputs:
    Audit_Eligibility_2022_vs_2026_Audit.xlsx
    Audit_Eligibility_2022_vs_2026_Summary.csv

Install: pip install pandas numpy yfinance openpyxl
Run: python audit_eligibility_2022_vs_2026.py

No portfolio re-selection or OOS performance estimation is performed here.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
INPUT = BASE / "candidate_tickers.csv"
OUTPUT = BASE / "audit_Eligibility_2022_vs_2026_Audit.xlsx"
SUMMARY_CSV = BASE / "audit_Eligibility_2022_vs_2026_Summary.csv"
START = "2012-05-01"
END_EXCLUSIVE = "2026-10-01"
BATCH = 50
TRAIN_END = pd.Timestamp("2022-05-31")
FULL_END = pd.Timestamp("2026-09-30")
PRICES_INDEX = pd.date_range("2012-05-31", FULL_END, freq="ME")
TRAIN_PRICE_INDEX = PRICES_INDEX[PRICES_INDEX <= TRAIN_END]
FULL_RETURN_INDEX = pd.date_range("2012-06-30", FULL_END, freq="ME")
TRAIN_RETURN_INDEX = FULL_RETURN_INDEX[FULL_RETURN_INDEX <= TRAIN_END]


def download_monthly_prices(tickers):
    chunks = []
    for start in range(0, len(tickers), BATCH):
        names = tickers[start:start+BATCH]
        print(f"Downloading {start+1}–{start+len(names)} / {len(tickers)}")
        try:
            raw = yf.download(
                names, start=START, end=END_EXCLUSIVE, auto_adjust=True,
                progress=False, threads=True, group_by="column",
            )
            if raw.empty:
                block = pd.DataFrame(index=pd.DatetimeIndex([]), columns=names, dtype=float)
            else:
                if isinstance(raw.columns, pd.MultiIndex):
                    block = raw["Close"]
                else:
                    block = raw[["Close"]].rename(columns={"Close": names[0]})
                if isinstance(block, pd.Series):
                    block = block.to_frame(name=names[0])
                block = block.reindex(columns=names)
                block.index = pd.to_datetime(block.index)
                if block.index.tz is not None:
                    block.index = block.index.tz_localize(None)
                block = block.resample("ME").last()
        except Exception as exc:
            print(f"WARNING: batch download failed: {exc}")
            block = pd.DataFrame(index=pd.DatetimeIndex([]), columns=names, dtype=float)
        chunks.append(block)
    return pd.concat(chunks, axis=1).reindex(PRICES_INDEX).reindex(columns=tickers)


def complete_window(prices, returns, price_dates, return_dates):
    p = prices.reindex(price_dates)
    r = returns.reindex(return_dates)
    return bool(
        p.notna().all() and r.notna().all()
        and np.isfinite(p.to_numpy(dtype=float)).all()
        and np.isfinite(r.to_numpy(dtype=float)).all()
        and (p > 0).all()
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing {INPUT}; run build_historical_candidates.py first.")
    tickers = (pd.read_csv(INPUT, dtype=str)["Ticker"].dropna()
               .str.strip().str.upper().drop_duplicates().tolist())
    prices = download_monthly_prices(tickers)
    returns = prices.pct_change(fill_method=None).reindex(FULL_RETURN_INDEX)
    records = []
    for ticker in tickers:
        p, r = prices[ticker], returns[ticker]
        train_ok = complete_window(p, r, TRAIN_PRICE_INDEX, TRAIN_RETURN_INDEX)
        full_ok = complete_window(p, r, PRICES_INDEX, FULL_RETURN_INDEX)
        if full_ok and not train_ok:
            raise AssertionError("Full-window eligibility cannot exceed training eligibility.")
        records.append({
            "Ticker": ticker,
            "Eligible_Through_2022_05": train_ok,
            "Eligible_Through_2026_09": full_ok,
            "Training_Eligible_But_Not_Full": train_ok and not full_ok,
            "Missing_Training_Prices": int(p.reindex(TRAIN_PRICE_INDEX).isna().sum()),
            "Missing_OOS_Prices": int(p.loc[p.index > TRAIN_END].isna().sum()),
            "Missing_Training_Returns": int(r.reindex(TRAIN_RETURN_INDEX).isna().sum()),
            "Missing_OOS_Returns": int(r.loc[r.index > TRAIN_END].isna().sum()),
            "First_Available_Month": p.first_valid_index(),
            "Last_Available_Month": p.last_valid_index(),
        })
    audit = pd.DataFrame(records)
    counts = {
        "Historical candidate tickers": len(tickers),
        "Eligible through 2022-05 (training only)": int(audit["Eligible_Through_2022_05"].sum()),
        "Eligible through 2026-09 (full history)": int(audit["Eligible_Through_2026_09"].sum()),
        "Training eligible but excluded by full-history rule": int(audit["Training_Eligible_But_Not_Full"].sum()),
        "Ineligible during training": int((~audit["Eligible_Through_2022_05"]).sum()),
        "Training months": len(TRAIN_RETURN_INDEX),
        "OOS months": len(FULL_RETURN_INDEX) - len(TRAIN_RETURN_INDEX),
    }
    summary = pd.DataFrame(list(counts.items()), columns=["Metric", "Value"])
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        audit.to_excel(writer, sheet_name="Ticker_Audit", index=False)
        audit.loc[audit["Eligible_Through_2022_05"], ["Ticker"]].to_excel(
            writer, sheet_name="Training_Eligible", index=False)
        audit.loc[audit["Eligible_Through_2026_09"], ["Ticker"]].to_excel(
            writer, sheet_name="Full_History_Eligible", index=False)
        audit.loc[audit["Training_Eligible_But_Not_Full"]].to_excel(
            writer, sheet_name="Excluded_By_Future", index=False)
        notes = pd.DataFrame({"Interpretation": [
            "Training-only eligibility uses May 2012–May 2022 monthly prices and June 2012–May 2022 returns.",
            "Full-history eligibility additionally requires June 2022–September 2026 observations.",
            "The difference measures the number excluded by the future-completeness rule, NOT the causal size of survivorship bias.",
            "Missing Yahoo data may reflect ticker changes, acquisitions, delistings or provider gaps; investigate reasons separately.",
            "No delisting returns are imputed and no OOS performance is inferred.",
            "Historical S&P 500 membership and symbol mapping require independent validation.",
        ]})
        notes.to_excel(writer, sheet_name="Interpretation", index=False)
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = __import__("openpyxl").styles.Font(color="FFFFFF", bold=True)
                cell.fill = __import__("openpyxl").styles.PatternFill("solid", fgColor="183153")
            for col in ws.columns:
                letter = col[0].column_letter
                width = min(78, max(17, max(len(str(c.value or "")) for c in col[:100]) + 2))
                ws.column_dimensions[letter].width = width
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT}\nSaved: {SUMMARY_CSV}")
    if counts["Eligible through 2026-09 (full history)"] != 337:
        print("WARNING: full-history count differs from earlier 337; compare data versions and download results.")


if __name__ == "__main__":
    main()
