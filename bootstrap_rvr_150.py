"""Circular block bootstrap for 150 GMVP portfolios (monthly OOS returns).

Input: OOS_Returns_150_Portfolios.csv, placed next to this script.
Output: Bootstrap_RVR_150_Results.csv, placed next to this script.

The same resampled month indices are applied to all portfolios and methods,
preserving contemporaneous cross-portfolio dependence. Percentile intervals
quantify sampling uncertainty in the class-average portfolio RVR statistic.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'Robustness_150_Results/OOS_Returns_150_Portfolios.csv'
OUTPUT = BASE / 'Bootstrap_RVR_150_Reproduced.csv'
METHODS = ['Var', 'Diag', 'Corr', 'SF', 'NLS']
TIERS = ['High', 'Medium', 'Low']
B = 3000
BLOCK_LENGTH = 6
SEED = 20261021  # Fixed for reproducibility; original bootstrap seed was not recorded.


def rvr(sample, alternative):
    """RVR (%) per portfolio, using sample variance with ddof=1."""
    sample_var = np.var(sample, axis=-1, ddof=1)
    alt_var = np.var(alternative, axis=-1, ddof=1)
    if np.any(sample_var <= 0):
        raise ValueError('Non-positive Sample variance encountered.')
    return 100.0 * (1.0 - alt_var / sample_var[..., None])


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Input CSV not found: {INPUT}')
    df = pd.read_csv(INPUT, parse_dates=['Date'])
    required = {'Portfolio_ID', 'Tier', 'Date', 'Sample', *METHODS}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(df.columns))}')
    if df.duplicated(['Portfolio_ID', 'Date']).any():
        raise ValueError('Duplicate portfolio-month records found.')
    if df[list(required - {'Portfolio_ID', 'Tier', 'Date'})].isna().any().any():
        raise ValueError('Missing monthly returns found.')

    # Arrange data as [portfolio, month, method], with identical calendar months.
    portfolios = sorted(df['Portfolio_ID'].unique())
    dates = pd.DatetimeIndex(sorted(df['Date'].unique()))
    if len(portfolios) != 150 or len(dates) != 52:
        raise ValueError(f'Expected 150 portfolios and 52 months; got {len(portfolios)} and {len(dates)}.')
    names = ['Sample'] + METHODS
    arrays = np.empty((len(portfolios), len(dates), len(names)), dtype=float)
    labels = []
    for i, pid in enumerate(portfolios):
        part = df.loc[df['Portfolio_ID'] == pid].sort_values('Date')
        if len(part) != len(dates) or not np.array_equal(part['Date'].to_numpy(), dates.to_numpy()):
            raise ValueError(f'Missing or mismatched months for {pid}.')
        if part['Tier'].nunique() != 1:
            raise ValueError(f'Multiple tiers for {pid}.')
        labels.append(part['Tier'].iloc[0])
        arrays[i] = part[names].to_numpy(dtype=float)
    labels = np.asarray(labels)
    if set(labels) != set(TIERS) or any(np.sum(labels == tier) != 50 for tier in TIERS):
        raise ValueError('Expected exactly 50 portfolios in each tier.')

    n_months = len(dates)
    rng = np.random.default_rng(SEED)
    observed = rvr(arrays[:, :, 0], arrays[:, :, 1:].transpose(0, 2, 1))
    # observed shape: portfolios x 5 methods
    bootstrap_means = {tier: np.empty((B, len(METHODS))) for tier in TIERS}
    for b in range(B):
        # Circular blocks: draw start points, concatenate blocks, truncate to 52 months.
        starts = rng.integers(0, n_months, size=int(np.ceil(n_months / BLOCK_LENGTH)))
        indices = np.concatenate([(s + np.arange(BLOCK_LENGTH)) % n_months for s in starts])[:n_months]
        sample = arrays[:, indices, 0]
        alternative = arrays[:, indices, 1:].transpose(0, 2, 1)
        values = rvr(sample, alternative)
        for tier in TIERS:
            bootstrap_means[tier][b] = values[labels == tier].mean(axis=0)

    rows = []
    for tier in TIERS:
        subset = observed[labels == tier]
        for j, method in enumerate(METHODS):
            lo, hi = np.percentile(bootstrap_means[tier][:, j], [2.5, 97.5])
            rows.append({
                'Tier': tier,
                'Method': method,
                'Mean_RVR_Percent': float(subset[:, j].mean()),
                'Median_RVR_Percent': float(np.median(subset[:, j])),
                'Positive_RVR_Percent': float(100 * np.mean(subset[:, j] > 0)),
                'CI_95_Lower': float(lo),
                'CI_95_Upper': float(hi),
                'Bootstrap_Replicates': B,
                'Block_Length_Months': BLOCK_LENGTH,
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
    print(result.round(4).to_string(index=False))
    print(f'\nSaved: {OUTPUT}')
    print('Note: Exact CI bounds can differ from a previous run if its random seed differs.')


if __name__ == '__main__':
    main()
