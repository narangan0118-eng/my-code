"""Select six GMVP portfolios from 3000 random candidates (monthly returns).
HAC retained; Holm correction and bootstrap confidence intervals removed.
Inputs: Companies/Candidate_Universe_Sensitivity.xlsx, monthly.xlsx,
study_month_rolling.py. Output: Six_Portfolios_3000_HAC.xlsx.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import statsmodels.api as sm
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

BASE = Path(__file__).resolve().parent
UNIVERSE_FILE = BASE / 'Companies/Eligible_337_Monthly_Returns.xlsx'
ESTIMATOR_FILE = BASE / 'study_all_portfolios_rolling.py'
MARKET_FILE = BASE / 'monthly.xlsx'
OUTPUT = BASE / 'Six_Portfolios_5000_HAC.xlsx'
METHODS = ['Sample', 'Var', 'Diag', 'Corr', 'SF', 'NLS']
ALTERNATIVES = METHODS[1:]
WINDOW = 120
VALIDATION_MONTHS = 26
N_MIN, N_MAX = 10, 10  # All selected portfolios satisfy N/T < 0.1.
N_CANDIDATES = 5000
VALIDATION_STRIDE = 3  # Selection-stage subsampling ONLY; final OOS evaluates every month.
SEED = 20261023
HAC_LAGS = 3
RF_MONTHLY = 0.0  # Excess-return Sharpe; adjust if using a risk-free rate.


def load_estimator():
    if not ESTIMATOR_FILE.exists():
        raise FileNotFoundError(f'Missing original estimator script: {ESTIMATOR_FILE}')
    spec = importlib.util.spec_from_file_location('original_gmvp', ESTIMATOR_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ('estimate_covariances', 'solve_long_only_gmvp'):
        if not hasattr(mod, name):
            raise AttributeError(f'{ESTIMATOR_FILE.name}: {name} not found')
    return mod


def load_data():
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f'Missing {UNIVERSE_FILE}')
    data = pd.read_excel(UNIVERSE_FILE, sheet_name='Complete_Returns', index_col=0)
    data.index = pd.to_datetime(data.index).to_period('M')
    data = data.apply(pd.to_numeric, errors='raise')
    if data.index.has_duplicates or data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError('Incomplete or duplicated stock returns')
    if len(data) != 172 or data.shape[1] != 337:
        raise ValueError(f'Expected 172 months x 337 stocks; got {data.shape}')
    if not MARKET_FILE.exists():
        raise FileNotFoundError('monthly.xlsx is needed for the original SF estimator (SP500 prices).')
    market = pd.read_excel(MARKET_FILE, sheet_name='close', thousands=',')
    if not {'Date', 'SP500'}.issubset(market.columns):
        raise ValueError('monthly.xlsx/close requires Date and SP500 columns')
    market['Date'] = pd.to_datetime(market['Date']).dt.to_period('M')
    market = market.set_index('Date').sort_index()
    if market.index.has_duplicates:
        raise ValueError('Duplicate SP500 months')
    sp = pd.to_numeric(market['SP500'], errors='coerce')
    market_ret = sp.pct_change(fill_method=None).reindex(data.index)
    if market_ret.isna().any() or not np.isfinite(market_ret.to_numpy()).all():
        raise ValueError('SP500 monthly returns do not match the 172 stock-return months')
    if len(data) <= WINDOW + VALIDATION_MONTHS:
        raise ValueError('No independent test months')
    return data, market_ret


def run_backtest(tickers, data, market, mod, positions):
    x = data.loc[:, tickers].to_numpy(dtype=float)
    r = market.to_numpy(dtype=float)
    records, weights = [], {m: [] for m in METHODS}
    for t in positions:
        matrices, _ = mod.estimate_covariances(x[t-WINDOW:t], r[t-WINDOW:t])
        realized = x[t]
        record = {'Date': str(data.index[t])}
        for m in METHODS:
            w = np.asarray(mod.solve_long_only_gmvp(matrices[m]), dtype=float)
            record[m] = float(w @ realized)
            weights[m].append(w)
        records.append(record)
    return pd.DataFrame(records).set_index('Date'), {m: np.vstack(v) for m,v in weights.items()}


def rvr_values(oos):
    base = float(oos['Sample'].var(ddof=1))
    if base <= 1e-16:
        raise ValueError('Near-zero Sample OOS variance')
    return {m: 100*(1-float(oos[m].var(ddof=1))/base) for m in ALTERNATIVES}


def candidate_pool(columns, train, rng):
    """Fixed-seed nested candidate prefixes; balanced sizes N=5,...,11."""
    cols = np.asarray(columns)
    sizes = np.arange(N_MIN, N_MAX + 1)
    selected, seen = [], set()
    for k in range(N_CANDIDATES):
        n = int(sizes[k % len(sizes)])
        for attempt in range(10000):
            ix = tuple(sorted(rng.choice(len(cols), size=n, replace=False).tolist()))
            if ix not in seen:
                seen.add(ix)
                selected.append(tuple(cols[list(ix)]))
                break
        else:
            raise RuntimeError('Unable to find a new candidate')
    return selected


def portfolio_metrics(oos, weights, tickers, method):
    series = oos[method].to_numpy()
    vol = float(np.std(series, ddof=1))
    sharpe = float(np.sqrt(12)*(np.mean(series)-RF_MONTHLY)/vol) if vol > 0 else np.nan
    W = weights[method]
    # One-way turnover, measured from drifted previous weights before rebalancing.
    # Realized portfolio returns alone do not give per-asset returns; compute outside.
    hhi = float(np.mean(np.sum(W**2, axis=1)))
    return vol, sharpe, hhi


def turnover_drifted(weights, tickers, data, positions):
    X = data.loc[:, tickers].to_numpy()
    values = {}
    for m, W in weights.items():
        amounts = []
        for j in range(1, len(positions)):
            prev_t = positions[j-1]
            current_t = positions[j]
            if current_t != prev_t+1:
                continue
            drifted = W[j-1] * (1 + X[prev_t])
            drifted = drifted / drifted.sum()
            amounts.append(0.5*np.abs(W[j]-drifted).sum())
        values[m] = float(np.mean(amounts)) if amounts else np.nan
    return values


def hac_pvalue(oos, method):
    # Difference of centered squared returns: variance-focused paired loss.
    a = oos['Sample'].to_numpy()
    b = oos[method].to_numpy()
    diff = (a-a.mean())**2 - (b-b.mean())**2
    if np.std(diff) < 1e-14:
        return 1.0
    fit = sm.OLS(diff, np.ones((len(diff),1))).fit(cov_type='HAC', cov_kwds={'maxlags':HAC_LAGS})
    # One-sided alternative: Sample variance > shrinkage variance.
    from scipy.stats import norm
    z = float(fit.tvalues[0])
    return float(norm.sf(z)) if np.isfinite(z) else 1.0


def format_workbook(path, perf_cols, weight_cols):
    from openpyxl import load_workbook
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = 'C2' if ws.title == 'Portfolio_Weights' else 'C2'
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 32
        for cell in ws[1]:
            cell.fill = PatternFill('solid', fgColor='193653')
            cell.font = Font(bold=True, color='FFFFFF')
            cell.alignment = Alignment(horizontal='center',vertical='center',wrap_text=True)
        for i,col in enumerate(ws.columns,1):
            letter=get_column_letter(i)
            if ws.title=='Portfolio_Performance':
                ws.column_dimensions[letter].width = 65 if i==len(perf_cols) else (23 if i<=2 else 19)
            else:
                ws.column_dimensions[letter].width = 20 if i==1 else 15
            if ws.title=='Portfolio_Weights' and i>1:
                for c in list(col)[1:]:
                    if isinstance(c.value,(int,float)):
                        c.number_format='0.00"%"'
        for j in range(2,ws.max_row+1):
            if j%2==0:
                for c in ws[j]:
                    c.fill=PatternFill('solid',fgColor='EDF3F8')
    wb.save(path)


def main():
    if not (2 <= N_MIN <= N_MAX < WINDOW):
        raise ValueError('Require 2 <= N_MIN <= N_MAX < WINDOW for this original NLS implementation')
    rng = np.random.default_rng(SEED)
    mod = load_estimator()
    data, market = load_data()
    train = data.iloc[:WINDOW]
    candidates = candidate_pool(data.columns, train, rng)
    print('Candidate sizes:', pd.Series([len(c) for c in candidates]).value_counts().sort_index().to_dict())
    selection_positions = list(range(WINDOW, WINDOW+VALIDATION_MONTHS, VALIDATION_STRIDE))
    test_positions = list(range(WINDOW+VALIDATION_MONTHS, len(data)))
    print(f'{len(candidates)} candidates; {len(selection_positions)} validation dates; {len(test_positions)} untouched test months')
    best = {f'P{i}':(-np.inf,None) for i in range(6)}
    failures = 0
    for k,tickers in enumerate(candidates,1):
        try:
            oos,_ = run_backtest(tickers,data,market,mod,selection_positions)
            rvr = rvr_values(oos)
        except (ValueError,RuntimeError,ArithmeticError,np.linalg.LinAlgError) as exc:
            failures+=1
            print(f'Candidate {k}: skipped ({exc})')
            continue
        scores = [min(rvr.values())]+[rvr[m] for m in ALTERNATIVES]
        for i,score in enumerate(scores):
            pid=f'P{i}'
            if score>best[pid][0]:
                best[pid]=(score,tickers)
        print(f'Candidate {k}/{len(candidates)} | N={len(tickers)} | best min-RVR={best["P0"][0]:.3f}%')
    if any(tickers is None for _,tickers in best.values()):
        raise RuntimeError('No candidate completed; check inputs and optimizer')
    print('Selection finished. Starting completely untouched monthly test...')
    rows=[]
    weight_cols={}
    for i in range(6):
        pid=f'P{i}'
        objective='All' if i==0 else ALTERNATIVES[i-1]
        score,tickers=best[pid]
        oos,weights=run_backtest(tickers,data,market,mod,test_positions)
        rvr=rvr_values(oos)
        turnovers=turnover_drifted(weights,tickers,data,test_positions)
        avg_corr=float((data.loc[:,tickers].iloc[:WINDOW].corr().to_numpy().sum()-len(tickers))/(len(tickers)*(len(tickers)-1)))
        for m in METHODS:
            vol,sharpe,hhi=portfolio_metrics(oos,weights,tickers,m)
            p = hac_pvalue(oos,m) if m!='Sample' else np.nan
            mean_return_pct = 100 * float(oos[m].mean())
            risk_pct = 100 * vol  # Monthly OOS standard deviation; equals OOS volatility.
            rows.append({'Portfolio':pid,'Selection_Target':objective,'Estimator':m,'N':len(tickers),
                         'N/T':len(tickers)/WINDOW,'Mean_Correlation_Train':avg_corr,
                         'RVR_Percent':0.0 if m=='Sample' else rvr[m],
                         'OOS_Mean_Return_Monthly_Percent':mean_return_pct,
                         'OOS_Risk_Monthly_Percent':risk_pct,
                         'OOS_Volatility_Monthly_Percent':100*vol,'Sharpe_Annualized':sharpe,
                         'Turnover_One_Way':turnovers[m],'HHI_Mean':hhi,
                         'HAC_p_one_sided':p,
                         'Validation_Target_RVR_Percent':score,
                         'Tickers':', '.join(tickers)})
            weight_cols[f'{pid}_{m}']=pd.Series(weights[m][-1]*100,index=tickers)
    perf=pd.DataFrame(rows)
    weights_table=pd.DataFrame(weight_cols).sort_index()
    # Missing means not a constituent; numeric zero means selected but zero allocation.
    weights_table.index.name='Ticker'
    with pd.ExcelWriter(OUTPUT,engine='openpyxl') as writer:
        perf.to_excel(writer,sheet_name='Portfolio_Performance',index=False)
        weights_table.to_excel(writer,sheet_name='Portfolio_Weights',na_rep='—')
    format_workbook(OUTPUT,list(perf.columns),list(weights_table.columns))
    print('Saved:',OUTPUT)
    print('Important: selection is approximate; validation dates are subsampled, test months are complete.')
    print('Selection universes filtered using complete future histories: survivorship/look-ahead eligibility caveat remains.')
    print('P0 has all-positive RVR on test?', all(perf.loc[(perf.Portfolio=='P0')&(perf.Estimator!='Sample'),'RVR_Percent']>0))

if __name__=='__main__':
    main()
