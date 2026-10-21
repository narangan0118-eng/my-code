"""Candidate-universe sensitivity for three 10-stock correlation portfolios.

Install: pip install numpy pandas yfinance openpyxl matplotlib
Run build_historical_candidates.py first to create candidate_tickers.csv.
IMPORTANT: Source the universe from a documented HISTORICAL constituent list.
Do not infer historical S&P 500 membership from today's members.
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CANDIDATES_FILE = BASE / 'candidate_tickers.csv'
OUTPUT = BASE / 'Candidate_Universe_Sensitivity.xlsx'
PLOT = BASE / 'Candidate_Universe_Sensitivity.png'
START = '2012-05-01'  # May close is the base for June 2012 return
END_EXCLUSIVE = '2026-10-01'
REQUIRED_MONTHS = pd.date_range('2012-06-30', '2026-09-30', freq='ME')
K_VALUES = [20, 30, 50, 75, 100, 150, 200]
N_ASSETS = 10
TRAIN_MONTHS = 120
REPEATS = 100
SEARCH_TRIALS = 2000  # approximate Monte Carlo optimization per replicate
SEED = 20261019
DOWNLOAD_BATCH = 50


def load_tickers():
    if not CANDIDATES_FILE.is_file():
        raise FileNotFoundError(
            f'Missing {CANDIDATES_FILE}. Create CSV with a Ticker column '
            'containing historically eligible candidate symbols.'
        )
    df = pd.read_csv(CANDIDATES_FILE, dtype=str)
    if 'Ticker' not in df.columns:
        raise ValueError('candidate_tickers.csv must contain a Ticker column.')
    tickers = list(dict.fromkeys(df['Ticker'].dropna().str.strip().str.upper()))
    tickers = [x for x in tickers if x]
    if len(tickers) < min(K_VALUES):
        raise ValueError('Not enough candidate tickers.')
    return tickers


def fetch_prices(tickers):
    blocks = []
    for offset in range(0, len(tickers), DOWNLOAD_BATCH):
        batch = tickers[offset:offset + DOWNLOAD_BATCH]
        print(f'Downloading {offset+1}-{offset+len(batch)} / {len(tickers)}')
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
    prices = pd.concat(blocks, axis=1)
    required_prices = pd.date_range('2012-05-31', '2026-09-30', freq='ME')
    return prices.reindex(required_prices).reindex(columns=tickers)


def check_eligibility(prices, tickers):
    # pct_change(fill_method=None) NEVER fills missing prices automatically.
    returns = prices.pct_change(fill_method=None).reindex(REQUIRED_MONTHS)
    records = []
    eligible = []
    for ticker in tickers:
        p = prices[ticker]
        r = returns[ticker]
        missing_price = int(p.isna().sum())
        missing_return = int(r.isna().sum())
        # A zero or negative adjusted price is invalid.
        invalid_price = int((p.notna() & (p <= 0)).sum())
        ok = missing_price == 0 and missing_return == 0 and invalid_price == 0
        if ok:
            eligible.append(ticker)
        records.append({
            'Ticker': ticker,
            'Eligible': ok,
            'Missing_Monthly_Prices_From_May_2012': missing_price,
            'Missing_Required_Monthly_Returns': missing_return,
            'Nonpositive_Prices': invalid_price,
            'First_Available_Month': p.first_valid_index(),
            'Last_Available_Month': p.last_valid_index(),
            'Reason': 'Complete' if ok else 'Missing/invalid price or return'
        })
    return returns, eligible, pd.DataFrame(records)


def mean_corr_for_indices(indices, matrix):
    sub = matrix[np.ix_(indices, indices)]
    n = len(indices)
    return float((sub.sum() - n) / (n * (n - 1)))


def select_three(corr, rng):
    n = len(corr)
    seen = set()
    # Include deterministic first candidate and stochastic candidates.
    for _ in range(SEARCH_TRIALS):
        inds = tuple(sorted(rng.choice(n, N_ASSETS, replace=False).tolist()))
        seen.add(inds)
    scored = sorted((mean_corr_for_indices(ix, corr), ix) for ix in seen)
    middle = scored[len(scored) // 2]
    return {'Low': scored[0], 'Medium': middle, 'High': scored[-1]}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    tickers = load_tickers()
    print('Candidate tickers:', len(tickers))
    prices = fetch_prices(tickers)
    returns, eligible, eligibility = check_eligibility(prices, tickers)
    print('Eligible with all 173 monthly prices / 172 returns:', len(eligible))
    print('Excluded:', len(tickers) - len(eligible))
    if len(eligible) < min(K_VALUES):
        raise ValueError('Fewer than 20 eligible companies. Review Eligibility results.')

    # No OOS values enter correlation estimates or selection.
    train = returns[eligible].iloc[:TRAIN_MONTHS]
    if train.isna().any().any():
        raise AssertionError('Training data must be complete.')
    if TRAIN_MONTHS >= len(REQUIRED_MONTHS):
        raise ValueError('No out-of-sample months remain.')
    corr_full = train.corr()
    rng = np.random.default_rng(SEED)
    rows, selected_rows = [], []
    previous = {}
    for k in K_VALUES:
        if k > len(eligible):
            print(f'Skipping K={k}: only {len(eligible)} eligible companies')
            continue
        for repeat in range(1, REPEATS + 1):
            # Each replicate samples K eligible names, independently of OOS data.
            universe = sorted(rng.choice(eligible, size=k, replace=False).tolist())
            subcorr = corr_full.loc[universe, universe].to_numpy()
            choices = select_three(subcorr, rng)
            gap = choices['High'][0] - choices['Low'][0]
            for label, (score, indices) in choices.items():
                members = [universe[i] for i in indices]
                cov = train[members].cov().to_numpy()
                condition = float(np.linalg.cond(cov))
                eigmin = float(np.linalg.eigvalsh(cov)[0])
                key = (k, label)
                # Consecutive-run Jaccard describes instability across random
                # candidate universes AND approximate portfolio searches.
                stability = (jaccard(previous[key], members)
                             if key in previous else np.nan)
                previous[key] = members
                rows.append({
                    'K': k, 'Repeat': repeat, 'Portfolio': label,
                    'Mean_Correlation': score, 'High_Low_Gap': gap,
                    'Cov_Condition_Number': condition,
                    'Smallest_Eigenvalue': eigmin,
                    'Jaccard_With_Previous_Repeat': stability,
                })
                selected_rows.append({
                    'K': k, 'Repeat': repeat, 'Portfolio': label,
                    'Tickers': ', '.join(members)
                })
        print(f'Finished K={k}')

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError('No K values were feasible.')
    gaps = (results.drop_duplicates(['K', 'Repeat'])
            .groupby('K')['High_Low_Gap']
            .agg(['mean', 'std', 'min', 'max']).reset_index())
    gaps.columns = ['K', 'Gap_Mean', 'Gap_SD', 'Gap_Min', 'Gap_Max']
    metrics = (results.groupby(['K', 'Portfolio'], as_index=False)
               .agg(Mean_Correlation=('Mean_Correlation', 'mean'),
                    Correlation_SD=('Mean_Correlation', 'std'),
                    Mean_Condition_Number=('Cov_Condition_Number', 'mean'),
                    Mean_Jaccard=('Jaccard_With_Previous_Repeat', 'mean')))
    config = pd.DataFrame([
        ('Start price month', '2012-05'),
        ('First required return', str(REQUIRED_MONTHS[0].date())),
        ('Last required return', str(REQUIRED_MONTHS[-1].date())),
        ('Required returns', len(REQUIRED_MONTHS)),
        ('Training months', TRAIN_MONTHS),
        ('Training end', str(REQUIRED_MONTHS[TRAIN_MONTHS-1].date())),
        ('OOS months excluded from selection', len(REQUIRED_MONTHS)-TRAIN_MONTHS),
        ('N assets per portfolio', N_ASSETS),
        ('Repeats per K', REPEATS),
        ('Monte Carlo candidates per portfolio search', SEARCH_TRIALS),
        ('Seed', SEED),
        ('Universe source', 'Historical S&P 500 May 2012 candidate_tickers.csv; verify externally'),
    ], columns=['Setting', 'Value'])

    with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
        config.to_excel(writer, sheet_name='Settings', index=False)
        eligibility.to_excel(writer, sheet_name='Eligibility', index=False)
        pd.DataFrame({'Eligible_Ticker': eligible}).to_excel(
            writer, sheet_name='Eligible_Tickers', index=False)
        gaps.to_excel(writer, sheet_name='Gap_Summary', index=False)
        metrics.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
        results.to_excel(writer, sheet_name='All_Repetitions', index=False)
        pd.DataFrame(selected_rows).to_excel(
            writer, sheet_name='Selected_Members', index=False)
        train.to_excel(writer, sheet_name='Training_Returns')
        # All 172 return months, to permit a separate OOS GMVP study later.
        returns[eligible].to_excel(writer, sheet_name='Complete_Returns')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(gaps['K'], gaps['Gap_Mean'], yerr=gaps['Gap_SD'],
                marker='o', capsize=4)
    ax.set_xlabel('Candidate universe size (K)')
    ax.set_ylabel('High minus low mean pairwise correlation')
    ax.set_title('Correlation separation by candidate universe size')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=180)
    plt.close(fig)
    print('\nSaved Excel:', OUTPUT)
    print('Saved figure:', PLOT)
    print('NOTE: No universal optimal K is asserted; inspect stability and gap.')
    print('NOTE: Requiring complete data through 2026 introduces survivorship bias.')


if __name__ == '__main__':
    main()
