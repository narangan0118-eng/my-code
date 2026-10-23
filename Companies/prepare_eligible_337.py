"""Build complete monthly-return universe without K-sensitivity analysis.
Requires candidate_tickers.csv from build_historical_candidates.py.
Install: pip install numpy pandas yfinance openpyxl
"""
from pathlib import Path
import pandas as pd
import yfinance as yf

BASE = Path(__file__).resolve().parent
CANDIDATES_FILE = BASE / 'candidate_tickers.csv'
OUTPUT = BASE / 'Eligible_337_Monthly_Returns.xlsx'
START, END_EXCLUSIVE = '2012-05-01', '2026-10-01'
PRICE_MONTHS = pd.date_range('2012-05-31', '2026-09-30', freq='ME')
RETURN_MONTHS = pd.date_range('2012-06-30', '2026-09-30', freq='ME')
TRAIN_MONTHS = 120
DOWNLOAD_BATCH = 50


def main():
    tickers = (pd.read_csv(CANDIDATES_FILE, dtype=str)['Ticker']
               .dropna().str.strip().str.upper().drop_duplicates().tolist())
    tickers = [t for t in tickers if t]
    blocks = []
    for start in range(0, len(tickers), DOWNLOAD_BATCH):
        batch = tickers[start:start + DOWNLOAD_BATCH]
        print(f'Downloading {start + 1}–{start + len(batch)} of {len(tickers)}')
        raw = yf.download(batch, start=START, end=END_EXCLUSIVE,
                          auto_adjust=True, progress=False, threads=True,
                          group_by='column')
        if raw.empty:
            close = pd.DataFrame(index=pd.DatetimeIndex([]), columns=batch, dtype=float)
        else:
            close = raw['Close']
            if isinstance(close, pd.Series):
                close = close.to_frame(name=batch[0])
            close = close.reindex(columns=batch)
            close.index = pd.to_datetime(close.index)
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            close = close.resample('ME').last()
        blocks.append(close)
    prices = pd.concat(blocks, axis=1).reindex(PRICE_MONTHS).reindex(columns=tickers)
    returns = prices.pct_change(fill_method=None).reindex(RETURN_MONTHS)
    rows, eligible = [], []
    for ticker in tickers:
        p, r = prices[ticker], returns[ticker]
        missing_prices = int(p.isna().sum())
        missing_returns = int(r.isna().sum())
        nonpositive_prices = int((p.notna() & (p <= 0)).sum())
        ok = missing_prices == missing_returns == nonpositive_prices == 0
        if ok:
            eligible.append(ticker)
        rows.append({'Ticker': ticker, 'Eligible': ok,
                     'Missing_Monthly_Prices': missing_prices,
                     'Missing_Monthly_Returns': missing_returns,
                     'Nonpositive_Prices': nonpositive_prices})
    print(f'Eligible: {len(eligible)} / {len(tickers)}')
    if len(eligible) < 30:
        raise ValueError('Fewer than 30 complete tickers.')
    with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='Eligibility', index=False)
        pd.DataFrame({'Eligible_Ticker': eligible}).to_excel(
            writer, sheet_name='Eligible_Tickers', index=False)
        returns[eligible].iloc[:TRAIN_MONTHS].to_excel(writer, sheet_name='Training_Returns')
        returns[eligible].to_excel(writer, sheet_name='Complete_Returns')
    print('Saved:', OUTPUT)
    print('Caution: filtering on completeness through 2026 may induce survivorship bias.')


if __name__ == '__main__':
    main()
