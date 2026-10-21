"""Create candidate_tickers.csv from a dated historical S&P 500 snapshot.
Run on an internet-connected machine: pip install pandas requests
"""
from pathlib import Path
import ast
import re
import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
URL = ('https://raw.githubusercontent.com/chinobing/'
       'historical_sp500_constituents/main/sp_500_historical_components.csv')
SNAPSHOT_DATE = pd.Timestamp('2012-05-31')


def parse_tickers(value):
    if pd.isna(value):
        return []
    s = str(value).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple, set)):
            return [str(x).strip() for x in val]
    except (SyntaxError, ValueError):
        pass
    return [x.strip().strip("'\"[] ") for x in re.split(r'[,;|\s]+', s) if x.strip()]


def main():
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    raw_path = BASE / 'historical_sp500_source.csv'
    raw_path.write_bytes(response.content)
    df = pd.read_csv(raw_path)
    print('Historical file columns:', list(df.columns))
    date_cols = [c for c in df.columns if 'date' in str(c).lower()]
    if not date_cols:
        date_cols = [df.columns[0]]
    dates = pd.to_datetime(df[date_cols[0]], errors='coerce')
    eligible_rows = df.loc[dates <= SNAPSHOT_DATE].copy()
    if eligible_rows.empty:
        raise ValueError('No historical snapshot at/before 2012-05-31.')
    idx = dates.loc[eligible_rows.index].idxmax()
    snapshot_date = dates.loc[idx]
    if (SNAPSHOT_DATE - snapshot_date).days > 45:
        raise ValueError(f'Historical snapshot too old: {snapshot_date.date()}')
    possible = [c for c in df.columns if c != date_cols[0]
                and any(w in str(c).lower() for w in ('ticker', 'constituent', 'symbol'))]
    col = possible[0] if possible else df.columns[1]
    tickers = parse_tickers(df.loc[idx, col])
    tickers = sorted(set(x.upper().replace('.', '-') for x in tickers if x))
    if len(tickers) < 400:
        raise ValueError(f'Only {len(tickers)} tickers parsed; check source columns and format.')
    out = BASE / 'candidate_tickers.csv'
    pd.DataFrame({'Ticker': tickers}).to_csv(out, index=False)
    print('Snapshot date:', snapshot_date.date())
    print('Candidate tickers:', len(tickers))
    print('Saved:', out)
    print('Historical membership and ticker renames require independent verification.')


if __name__ == '__main__':
    main()
