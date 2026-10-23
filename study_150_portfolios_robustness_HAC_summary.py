"""150-portfolio robustness analysis AND 52-month OOS return export; reuses the six estimators from the original HAC script.
Place this script, study_all_portfolios_rolling.py, the selection workbook and Eligible_337_Monthly_Returns.xlsx,
and monthly.xlsx in the same folder, then run this script in Positron.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
SELECTION = HERE / 'Companies/Robustness_150_Portfolio_Selection.xlsx'
RETURNS = HERE / 'Companies/Eligible_337_Monthly_Returns.xlsx'
MARKET = HERE / 'monthly.xlsx'
ORIGINAL = HERE / 'study_all_portfolios_rolling.py'
OUT = HERE / 'Robustness_150_Results'
OUT.mkdir(exist_ok=True)
WINDOW = 120
METHODS = ['Sample', 'Var', 'Diag', 'Corr', 'SF', 'NLS']


def load_original():
    if not ORIGINAL.exists():
        raise FileNotFoundError(f'Үндсэн судалгааны код олдсонгүй: {ORIGINAL}')
    spec = importlib.util.spec_from_file_location('original_gmvp', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data():
    for f in [SELECTION, RETURNS, MARKET]:
        if not f.exists():
            raise FileNotFoundError(f'Оролтын файл олдсонгүй: {f}')
    summary = pd.read_excel(SELECTION, sheet_name='Portfolio_Summary')
    members = pd.read_excel(SELECTION, sheet_name='Portfolio_Members')
    y = pd.read_excel(RETURNS, sheet_name='Complete_Returns', index_col=0)
    y.index = pd.to_datetime(y.index)
    if y.shape[0] != 172 or y.index.has_duplicates or y.isna().any().any():
        raise ValueError('Complete_Returns: 172 сар, дутуу утгагүй өгөгдөл шаардлагатай.')
    close = pd.read_excel(MARKET, sheet_name='close', thousands=',')
    if not {'Date', 'SP500'}.issubset(close.columns):
        raise ValueError('monthly.xlsx / close: Date, SP500 багана шаардлагатай.')
    close['Date'] = pd.to_datetime(close['Date'])
    close = close.set_index('Date').sort_index()
    sp = pd.to_numeric(close['SP500'], errors='coerce')
    if sp.isna().any() or (sp <= 0).any():
        raise ValueError('SP500-ийн үнэ дутуу эсвэл эерэг биш байна.')
    sp.index = sp.index.to_period('M')
    if sp.index.has_duplicates:
        raise ValueError('SP500 сарын огноо давхардсан байна.')
    market = sp.pct_change(fill_method=None).iloc[1:].reindex(y.index.to_period('M'))
    if market.isna().any():
        raise ValueError('SP500 өгөөжийн сарууд Complete_Returns-тай таарахгүй байна.')
    market.index = y.index
    required = set(members['Ticker'].astype(str))
    missing = sorted(required - set(y.columns.astype(str)))
    if missing:
        raise ValueError(f'Complete_Returns-д {len(missing)} ticker дутуу: {missing[:15]}')
    if len(summary) != 150 or len(members) != 1500:
        raise ValueError('150 багц, 1500 хувьцааны бүртгэл шаардлагатай.')
    return summary, members, y, market


def run_one(pid, tier, corr, tickers, all_returns, market, original):
    if len(tickers) != 10 or len(set(tickers)) != 10:
        raise ValueError(f'{pid}: 10 ялгаатай ticker байх ёстой.')
    if not set(tickers).issubset(all_returns.columns):
        raise ValueError(f'{pid}: өгөөжийн багана дутуу.')
    Y = all_returns[tickers].to_numpy(dtype=float)
    rm = market.to_numpy(dtype=float)
    dates = all_returns.index[WINDOW:]
    oos = np.empty((len(dates), len(METHODS)))
    weight_history = np.empty((len(dates), len(METHODS), 10))
    for j, t in enumerate(range(WINDOW, len(all_returns))):
        sigmas, _ = original.estimate_covariances(Y[t-WINDOW:t], rm[t-WINDOW:t])
        for k, method in enumerate(METHODS):
            w = original.solve_long_only_gmvp(sigmas[method])
            oos[j, k] = float(w @ Y[t])
            weight_history[j, k] = w
    variance = np.var(oos, axis=0, ddof=1)
    mean = np.mean(oos, axis=0)
    sample_var = variance[0]
    rows = []
    for k, method in enumerate(METHODS):
        W = weight_history[:, k, :]
        hhi = np.mean(np.sum(W * W, axis=1))
        turnovers = []
        for j in range(1, len(dates)):
            prev_r = Y[WINDOW+j-1]
            prev_w = W[j-1]
            drifted = prev_w * (1 + prev_r) / (1 + prev_w @ prev_r)
            turnovers.append(0.5 * np.abs(W[j] - drifted).sum())
        rows.append({
            'Portfolio_ID': pid, 'Tier': tier, 'Mean_Pairwise_Correlation': corr,
            'Method': method, 'Window': WINDOW, 'OOS_Months': len(dates),
            'Mean_Monthly_Return': mean[k], 'OOS_Monthly_Variance': variance[k],
            'OOS_Annual_Volatility': np.sqrt(12 * variance[k]),
            'RVR_Percent': 100 * (1 - variance[k] / sample_var),
            'Sharpe_Annual': np.sqrt(12) * mean[k] / np.sqrt(variance[k]),
            'Mean_Turnover': np.mean(turnovers), 'Mean_HHI': hhi
        })
    oos_frame = pd.DataFrame(oos, columns=METHODS)
    oos_frame.insert(0, "Date", dates.strftime("%Y-%m-%d"))
    oos_frame.insert(0, "Tier", tier)
    oos_frame.insert(0, "Portfolio_ID", pid)
    return pd.DataFrame(rows), oos_frame


def main():
    original = load_original()
    summary, members, Y, market = load_data()
    if list(Y.index[WINDOW:].to_period('M').astype(str)) != list(pd.period_range('2022-06','2026-09',freq='M').astype(str)):
        raise ValueError('OOS хугацаа 2022-06–2026-09 байх ёстой.')
    results = []
    oos_frames = []
    oos_dir = OUT / 'OOS_Returns_By_Portfolio'
    oos_dir.mkdir(exist_ok=True)
    dates = Y.index[WINDOW:]
    for i, row in enumerate(summary.itertuples(index=False), 1):
        pid = row.Portfolio_ID
        cached = OUT / f'{pid}.csv'
        oos_path = oos_dir / f'{pid}_OOS.csv'
        need_summary = not cached.exists()
        need_oos = not oos_path.exists()
        if need_summary or need_oos:
            tickers = members.loc[members['Portfolio_ID'] == pid, 'Ticker'].tolist()
            calculated_summary, calculated_oos = run_one(
                pid, row.Tier, row.Mean_Pairwise_Correlation, tickers, Y, market, original
            )
            if need_summary:
                calculated_summary.to_csv(cached, index=False)
            if need_oos:
                calculated_oos.to_csv(oos_path, index=False)
        one = pd.read_csv(cached)
        if len(one) != 6 or set(one['Method']) != set(METHODS):
            raise ValueError(f'Дутуу/буруу үр дүн: {cached}')
        oos_one = pd.read_csv(oos_path)
        if len(oos_one) != len(dates) or list(oos_one.columns) != ['Portfolio_ID', 'Tier', 'Date', *METHODS]:
            raise ValueError(f'OOS файл буруу бүтэцтэй: {oos_path}')
        if oos_one['Portfolio_ID'].nunique() != 1 or oos_one['Portfolio_ID'].iloc[0] != pid:
            raise ValueError(f'OOS Portfolio_ID буруу: {oos_path}')
        if list(pd.to_datetime(oos_one['Date']).dt.to_period('M')) != list(dates.to_period('M')):
            raise ValueError(f'OOS огноо таарахгүй: {oos_path}')
        for method in METHODS:
            computed = oos_one[method].var(ddof=1)
            expected = one.set_index('Method').loc[method, 'OOS_Monthly_Variance']
            if not np.isclose(computed, expected, rtol=1e-7, atol=1e-12):
                raise ValueError(f'{pid}/{method}: OOS дисперс өмнөх үр дүнтэй таарахгүй.')
        results.append(one)
        oos_frames.append(oos_one)
        print(f'{i:03d}/150  {pid}  summary + OOS бэлэн', flush=True)
    oos_all = pd.concat(oos_frames, ignore_index=True)
    if len(oos_all) != 7800 or oos_all.groupby('Portfolio_ID').size().ne(52).any():
        raise ValueError('150 багц x 52 сарын OOS өгөөж шаардлагатай.')
    oos_all.to_csv(OUT / 'OOS_Returns_150_Portfolios.csv', index=False)
    full = pd.concat(results, ignore_index=True)
    full.to_csv(OUT / 'Portfolio_Method_Results.csv', index=False)
    shrink = full.loc[full.Method != 'Sample'].copy()
    # HAC test for differences in centered OOS squared returns (variance).
    # Center each portfolio/method over its own 52-month OOS mean first.
    # Average differences across 50 portfolios for each month, then apply
    # Newey-West HAC to the resulting 52-month time series.
    # This tests the mean of portfolio-specific variance differences,
    # NOT the variance of an equally weighted aggregate portfolio.
    hac_rows = []
    for (tier, method), group in shrink.groupby(['Tier', 'Method'], sort=False):
        ids = group['Portfolio_ID'].astype(str).tolist()
        sub = oos_all.loc[
            (oos_all['Tier'] == tier) &
            (oos_all['Portfolio_ID'].astype(str).isin(ids))
        ].copy()
        if sub['Portfolio_ID'].nunique() != 50 or len(sub) != 50 * 52:
            raise ValueError(f'{tier}/{method}: 50 багц x 52 сарын OOS өгөгдөл шаардлагатай.')
        for col in ['Sample', method]:
            sub[col + '_centered'] = sub[col].astype(float) - sub.groupby('Portfolio_ID')[col].transform('mean')
        sub['Variance_Difference'] = (
            sub['Sample_centered'] ** 2 - sub[method + '_centered'] ** 2
        )
        monthly = sub.groupby('Date')['Variance_Difference'].mean().sort_index().to_numpy()
        if len(monthly) != 52 or not np.isfinite(monthly).all():
            raise ValueError(f'{tier}/{method}: HAC-д 52 сарын цуваа шаардлагатай.')
        # Check algebraically against mean of sample variances (ddof=1).
        expected_difference = (
            group['Portfolio_ID'].map(
                full.loc[full.Method == 'Sample'].set_index('Portfolio_ID')['OOS_Monthly_Variance']
            ).to_numpy() - group['OOS_Monthly_Variance'].to_numpy()
        ).mean()
        if not np.isclose(monthly.mean() * 52 / 51, expected_difference, rtol=1e-7, atol=1e-12):
            raise ValueError(f'{tier}/{method}: HAC variance difference does not match OOS variances.')
        n = len(monthly)
        residual = monthly - monthly.mean()
        lag = 3
        long_run_var = np.dot(residual, residual) / n
        for ell in range(1, lag + 1):
            gamma = np.dot(residual[ell:], residual[:-ell]) / n
            long_run_var += 2 * (1 - ell / (lag + 1)) * gamma
        se = np.sqrt(max(long_run_var, 0) / n)
        statistic = float(monthly.mean() / se) if se > 0 else np.nan
        p_value = float(2 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
        hac_rows.append({
            'Tier': tier, 'Method': method,
            'HAC_Mean_Variance_Difference': float(expected_difference),
            'HAC_Statistic': statistic, 'HAC_p_value': p_value,
            'HAC_Significant_5pct': bool(p_value < 0.05) if np.isfinite(p_value) else False,
            'HAC_Lags': lag
        })
    hac = pd.DataFrame(hac_rows)
    aggregated = shrink.groupby(['Tier', 'Method'], sort=False).agg(
        Portfolios=('RVR_Percent', 'size'),
        Mean_RVR=('RVR_Percent', 'mean'), Median_RVR=('RVR_Percent', 'median'),
        SD_RVR=('RVR_Percent', 'std'),
        Positive_RVR_Share=('RVR_Percent', lambda x: float(np.mean(x > 0))),
        Mean_Turnover=('Mean_Turnover', 'mean'), Mean_HHI=('Mean_HHI', 'mean'),
        Mean_Monthly_Return=('Mean_Monthly_Return', 'mean'),
        Mean_OOS_Monthly_Variance=('OOS_Monthly_Variance', 'mean'),
        Mean_OOS_Annual_Volatility=('OOS_Annual_Volatility', 'mean'),
        Mean_Sharpe_Annual=('Sharpe_Annual', 'mean')
    ).reset_index()
    aggregated = aggregated.merge(hac, on=['Tier', 'Method'], how='left', validate='one_to_one')
    aggregated.to_csv(OUT / 'Robustness_Summary.csv', index=False)
    print('\n=== Нэгдсэн үр дүн ===\n', aggregated.round(4).to_string(index=False))
    print('\nФайлууд:', OUT)
    print('OOS өгөөж: ', OUT / 'OOS_Returns_150_Portfolios.csv')
    print('900 OOS дисперсийг багц тус бүрээр тулгаж шалгасан.')
    print('Тайлбар: багцууд хувьцаа болон OOS хугацаагаа хуваалцдаг; бие даасан ажиглалт биш.')


if __name__ == '__main__':
    main()
