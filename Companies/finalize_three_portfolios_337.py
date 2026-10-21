"""Finalize three disjoint 10-stock portfolios using training returns only.
Input: Candidate_Universe_Sensitivity.xlsx (the 453/337-stock version).
Outputs: Final_Three_Portfolios.xlsx, final_portfolio_tickers.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'Candidate_Universe_Sensitivity.xlsx'
OUTPUT = BASE / 'Final_Three_Portfolios_337.xlsx'
CSV_OUTPUT = BASE / 'final_portfolio_tickers_337.csv'
N, TRAIN_MONTHS = 10, 120
SEARCH_SEED = 20261020
SEARCH_TRIALS = 100000
REQUIRE_DISJOINT = True


def avg_corr(matrix, indices):
    block = matrix[np.ix_(indices, indices)]
    return float((block.sum() - len(indices)) / (len(indices) * (len(indices) - 1)))


def find_portfolio(matrix, allowed, rng, goal, target=None):
    allowed = np.asarray(allowed, dtype=int)
    if len(allowed) < N:
        raise ValueError('Too few eligible tickers for portfolio selection.')
    best, best_loss = None, np.inf
    for _ in range(SEARCH_TRIALS):
        ix = np.sort(rng.choice(allowed, size=N, replace=False))
        score = avg_corr(matrix, ix)
        loss = -score if goal == 'High' else score if goal == 'Low' else abs(score - target)
        if loss < best_loss:
            best, best_loss = (score, ix), loss
    return best


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'Input not found: {INPUT}')
    eligible = pd.read_excel(INPUT, sheet_name='Eligible_Tickers')['Eligible_Ticker'].dropna().astype(str).tolist()
    training = pd.read_excel(INPUT, sheet_name='Training_Returns', index_col=0)
    complete = pd.read_excel(INPUT, sheet_name='Complete_Returns', index_col=0)
    training.index = pd.to_datetime(training.index)
    complete.index = pd.to_datetime(complete.index)
    if len(training) != TRAIN_MONTHS or len(complete) != 172:
        raise ValueError(f'Unexpected return lengths: training={len(training)}, full={len(complete)}')
    if len(eligible) != len(set(eligible)):
        raise ValueError('Duplicate tickers in Eligible_Tickers sheet.')
    if not set(eligible).issubset(training.columns) or not set(eligible).issubset(complete.columns):
        raise ValueError('Missing eligible tickers in return sheets.')
    if training[eligible].isna().any().any() or complete[eligible].isna().any().any():
        raise ValueError('Missing returns for eligible tickers.')
    if not np.allclose(training[eligible].to_numpy(), complete[eligible].iloc[:TRAIN_MONTHS].to_numpy()):
        raise ValueError('Training returns differ from first 120 complete returns.')
    universe = sorted(set(eligible))
    K = len(universe)
    if K < 3 * N:
        raise ValueError(f"Need at least {3*N} eligible stocks, found {K}")
    print(f"Using all {K} eligible stocks; no 100-stock subsampling.")
    corr = training[universe].corr().to_numpy()
    if not np.isfinite(corr).all():
        raise ValueError('Invalid training correlation matrix.')
    target = float((corr.sum() - K) / (K * (K - 1)))
    rng = np.random.default_rng(SEARCH_SEED)
    remaining = set(range(K))
    selections = {}
    for label in ('High', 'Low', 'Medium'):
        score, ix = find_portfolio(corr, sorted(remaining), rng, label, target)
        members = [universe[i] for i in ix]
        selections[label] = (score, members)
        if REQUIRE_DISJOINT:
            remaining.difference_update(ix)
        print(f'{label}: corr={score:.4f}, members={", ".join(members)}')
    summary, member_rows = [], []
    for label in ('High', 'Medium', 'Low'):
        score, members = selections[label]
        cov = training[members].cov().to_numpy()
        summary.append({'Portfolio': label, 'Mean_Pairwise_Correlation': score,
                        'Covariance_Condition_Number': float(np.linalg.cond(cov)),
                        'Smallest_Covariance_Eigenvalue': float(np.linalg.eigvalsh(cov)[0]),
                        'Tickers': ', '.join(members)})
        member_rows.extend({'Portfolio': label, 'Ticker': t} for t in members)
    members_df = pd.DataFrame(member_rows)
    config = pd.DataFrame([
        ('Candidate source', INPUT.name), ('Eligible stocks', len(eligible)),
        ('Universe K', K), ('Assets per portfolio N', N),
        ('Universe rule', 'All eligible tickers, alphabetically sorted'), ('Search seed', SEARCH_SEED),
        ('Search trials per portfolio', SEARCH_TRIALS),
        ('Disjoint portfolios', REQUIRE_DISJOINT),
        ('Training returns', TRAIN_MONTHS),
        ('Training first month', str(training.index[0].date())),
        ('Training last month', str(training.index[-1].date())),
        ('OOS first month', str(complete.index[TRAIN_MONTHS].date())),
        ('OOS last month', str(complete.index[-1].date())),
        ('Medium target', target),
        ('Selection method', 'Monte Carlo approximate, sequential High-Low-Medium'),
        ('Caveat', 'Eligibility through 2026 may induce survivorship/look-ahead bias'),
    ], columns=['Setting', 'Value'])
    with pd.ExcelWriter(OUTPUT, engine='openpyxl') as writer:
        config.to_excel(writer, sheet_name='Settings', index=False)
        pd.DataFrame({'Ticker': universe}).to_excel(writer, sheet_name='Candidate_All', index=False)
        pd.DataFrame(summary).to_excel(writer, sheet_name='Portfolio_Summary', index=False)
        members_df.to_excel(writer, sheet_name='Portfolio_Members', index=False)
        training[universe].to_excel(writer, sheet_name='Training_All')
        complete[[t for t in members_df['Ticker']]].to_excel(writer, sheet_name='Selected_Returns_172')
        for label in ('High', 'Medium', 'Low'):
            training[selections[label][1]].corr().to_excel(writer, sheet_name=f'Corr_{label}')
    members_df.to_csv(CSV_OUTPUT, index=False, encoding='utf-8-sig')
    print('Saved:', OUTPUT)
    print('Saved:', CSV_OUTPUT)

if __name__ == '__main__':
    main()
