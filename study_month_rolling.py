import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INPUT_FILE = Path("monthly.xlsx")
OUTPUT_FILE = "monthly_rolling_window.xlsx"
SHEET_NAME = "close"

WINDOW = 120 #сар бүрийн GMVP жинг тооцохдоо өмнөх 120 сарын өгөөжийг ашиглана.
STOCKS = ['Apple', 'Microsoft', 'Google', 'Amazon', 'Nvidia',
          'Meta', 'AMD', 'Tesla', 'Intel', 'Broadcom'] 
METHODS = ['Sample', 'Var', 'Diag', 'Corr', 'SF', 'NLS'] #Ковариацын матрицын зургаан үнэлэгчийн нэр


def load_returns(path):
    close = pd.read_excel(path,sheet_name=SHEET_NAME,thousands=',')
    close['Date'] = pd.to_datetime(close['Date'])
    close = close.set_index('Date').sort_index()
    close = close[STOCKS + ['SP500']].astype(float)
    if close.index.has_duplicates:
        raise ValueError("Давхардсан огноо байна.")
    if close.isna().any().any():
        raise ValueError("Үнийн өгөгдөлд хоосон утга байна.")
    returns = close.pct_change(fill_method=None).iloc[1:]
    return returns[STOCKS], returns['SP500']


def epanechnikov_kernel(x):
    out = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= np.sqrt(5.0)
    out[mask] = (3.0 / (4.0 * np.sqrt(5.0))) * (1.0 - x[mask] ** 2 / 5.0)
    return out


def hilbert_epanechnikov_kernel(x):
    # Same Hilbert-kernel expression as the user's existing study_month.py.
    out = np.zeros_like(x, dtype=float)
    boundary = np.isclose(np.abs(x), np.sqrt(5.0), atol=1e-12, rtol=0)
    inside = ~boundary
    xx = x[inside]
    with np.errstate(divide='ignore', invalid='ignore'):
        logarithm = np.log(np.abs((np.sqrt(5.0) - xx) / (np.sqrt(5.0) + xx)))
        out[inside] = (-3.0 * xx / (10.0 * np.pi)
                       + (3.0 / (4.0 * np.sqrt(5.0) * np.pi))
                       * (1.0 - xx ** 2 / 5.0) * logarithm)
    out[boundary] = -3.0 * x[boundary] / (10.0 * np.pi)
    return out


def estimate_covariances(stock_window, market_window):
    """Recalculate six covariance estimators using ONLY current training data."""
    Y = np.asarray(stock_window, dtype=float)
    rm = np.asarray(market_window, dtype=float)
    n, N = Y.shape
    X = Y - Y.mean(axis=0)
    S = X.T @ X / n  # ddof=0, consistent with original code
    Z = X[:, :, None] * X[:, None, :]
    diff = Z - S
    b_hat = np.sum(diff ** 2) / (n ** 2 * N)
    result = {'Sample': S}
    alphas = {}

    # 1. Constant variance target
    T_var = np.eye(N) * (np.trace(S) / N)
    d_var = np.sum((S - T_var) ** 2) / N
    alpha_var = float(np.clip(b_hat / d_var, 0, 1)) if d_var > 0 else 1.0
    result['Var'] = (1 - alpha_var) * S + alpha_var * T_var
    alphas['Var'] = alpha_var

    # 2. Diagonal target
    T_diag = np.diag(np.diag(S))
    diff_diag = diff.copy()
    diag_idx = np.arange(N)
    diff_diag[:, diag_idx, diag_idx] = 0.0
    b_diag = np.sum(diff_diag ** 2) / (n ** 2 * N)
    d_diag = np.sum((S - T_diag) ** 2) / N
    alpha_diag = float(np.clip(b_diag / d_diag, 0, 1)) if d_diag > 0 else 1.0
    result['Diag'] = (1 - alpha_diag) * S + alpha_diag * T_diag
    alphas['Diag'] = alpha_diag

    # 3. Constant correlation target, using original theta/c_hat formula
    sd = np.sqrt(np.diag(S))
    if np.any(sd <= 0):
        raise ValueError('Тогтмол өгөөжтэй хувьцаа сургалтын цонхонд байна.')
    R = S / np.outer(sd, sd)
    rbar = (np.sum(R) - np.trace(R)) / (N * (N - 1))
    T_corr = rbar * np.outer(sd, sd)
    np.fill_diagonal(T_corr, np.diag(S))
    d_corr = np.sum((S - T_corr) ** 2) / N
    var_y2 = np.mean((X ** 2 - np.diag(S)) ** 2, axis=0)
    theta_sum = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            offdiag = X[:, i] * X[:, j] - S[i, j]
            theta_ii_ij = np.mean((X[:, i] ** 2 - S[i, i]) * offdiag)
            theta_jj_ij = np.mean((X[:, j] ** 2 - S[j, j]) * offdiag)
            theta_sum += (np.sqrt(S[j, j] / S[i, i]) * theta_ii_ij
                          + np.sqrt(S[i, i] / S[j, j]) * theta_jj_ij)
    c_corr = (np.sum(var_y2) + rbar * theta_sum / 2) / (n * N)
    alpha_corr = float(np.clip((b_hat - c_corr) / d_corr, 0, 1)) if d_corr > 0 else 1.0
    result['Corr'] = (1 - alpha_corr) * S + alpha_corr * T_corr
    alphas['Corr'] = alpha_corr

    # 4. Single-factor target, preserving original shrinkage formula
    rm_centered = rm - rm.mean()
    sigma2_m = np.mean(rm_centered ** 2)
    if sigma2_m <= 0:
        raise ValueError('SP500 өгөөжийн вариац 0 байна.')
    beta = np.sum(X * rm_centered[:, None], axis=0) / np.sum(rm_centered ** 2)
    residual = X - np.outer(rm_centered, beta)
    sigma2_eps = np.mean(residual ** 2, axis=0)
    T_sf = np.outer(beta, beta) * sigma2_m + np.diag(sigma2_eps)
    d_sf = np.sum((S - T_sf) ** 2) / N
    q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            term2 = X[:, i] * X[:, j] - S[i, j]
            if i == j:
                term1 = X[:, i] ** 2 - S[i, i]
            else:
                term1 = (beta[j] * X[:, i] * rm_centered
                         + beta[i] * X[:, j] * rm_centered
                         - beta[i] * beta[j] * rm_centered ** 2 - T_sf[i, j])
            q[i, j] = np.mean(term1 * term2)
    c_sf = np.sum(q) / (n * N)
    alpha_sf = float(np.clip((b_hat - c_sf) / d_sf, 0, 1)) if d_sf > 0 else 1.0
    result['SF'] = (1 - alpha_sf) * S + alpha_sf * T_sf
    alphas['SF'] = alpha_sf

    # 5. Analytic nonlinear shrinkage: original kernel/Hilbert implementation
    eig, U = np.linalg.eigh(S)
    order = np.argsort(eig)[::-1]
    eig = np.maximum(eig[order], 1e-8)
    U = U[:, order]
    c = N / n
    h = n ** (-1.0 / 3.0)
    hj = eig * h
    f = np.empty(N)
    Hf = np.empty(N)
    for i in range(N):
        x = (eig[i] - eig) / hj
        f[i] = np.mean(epanechnikov_kernel(x) / hj)
        Hf[i] = np.mean(hilbert_epanechnikov_kernel(x) / hj)
    d = eig / ((1 - c - c * np.pi * eig * Hf) ** 2
               + (c * np.pi * eig * f) ** 2)
    result['NLS'] = (U * d) @ U.T

    for name, matrix in result.items():
        matrix = (matrix + matrix.T) / 2
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f'{name} матрицад NaN/Inf байна.')
        if np.linalg.eigvalsh(matrix).min() < -1e-10:
            raise ValueError(f'{name} матриц эерэг хагас тодорхой биш байна.')
        result[name] = matrix
    return result, alphas


def solve_long_only_gmvp(sigma):
    N = sigma.shape[0]
    sigma = (sigma + sigma.T) / 2
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0,
            'jac': lambda w: np.ones(N)}
    res = minimize(lambda w: w @ sigma @ w, np.full(N, 1.0 / N),
                   jac=lambda w: 2 * sigma @ w,
                   #Sequential Least Squares Programming аргаар хязгаарлалттай оновчлолын бодлогыг бодно.
                   method='SLSQP', bounds=[(0.0, 1.0)] * N,
                   constraints=[cons],
                   options={'ftol': 1e-15, 'maxiter': 10000})
    if not res.success:
        raise RuntimeError(f'GMVP оновчлол амжилтгүй: {res.message}')
    weights = np.where(np.abs(res.x) < 1e-10, 0.0, res.x)
    if abs(weights.sum() - 1.0) > 1e-7 or weights.min() < -1e-8:
        raise RuntimeError('GMVP жингийн хязгаарлалт зөрчигдлөө.')
    return weights
# ==================================================
# GMVP-ийн түүврээс гадуурх сарын өгөөжийн график
# ==================================================

def plot_oos_returns(oos):

    fig, ax = plt.subplots(figsize=(12, 6))

    for method in METHODS:
        ax.plot(
            oos.index,
            oos[method] * 100,
            label=method,
            linewidth=1.5
        )

    ax.axhline(
        y=0,
        color='black',
        linestyle='--',
        linewidth=0.8
    )

    ax.set_title(
        'Out-of-Sample Monthly Returns of GMVP',
        fontsize=13
    )

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Monthly Return (%)', fontsize=11)

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=6)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%Y-%m')
    )

    ax.tick_params(axis='x', rotation=45)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=6,
        frameon=False
    )

    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        'GMVP_OOS_Monthly_Returns.png',
        dpi=300,
        bbox_inches='tight'
    )

    fig.savefig(
        'GMVP_OOS_Monthly_Returns.pdf',
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)
# ==================================================
# Relative Variance Reduction (RVR) Graph
# ==================================================

def plot_rvr(rvr_percent):

    fig, ax = plt.subplots(figsize=(9, 5))

    values = rvr_percent.reindex(METHODS)

    bars = ax.bar(
        values.index,
        values.values,
        width=0.6,
        color=[
            '#808080' if v == 0
            else '#2E86AB' if v > 0
            else '#C44E52'
            for v in values.values
        ]
    )

    ax.axhline(0, color='black', linewidth=0.8)

    for bar, value in zip(bars, values.values):
        ax.annotate(
            f'{value:.2f}%',
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5 if value >= 0 else -15),
            textcoords='offset points',
            ha='center',
            fontsize=10
        )

    ax.set_ylabel('Relative Variance Reduction (%)')
    ax.set_xlabel('Covariance Estimator')

    ax.set_title(
        'Out-of-Sample Relative Variance Reduction',
        fontsize=12
    )

    ax.set_ylim(
        min(-2, values.min() - 1),
        max(12, values.max() + 1)
    )

    ax.grid(axis='y', alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()

    plt.savefig(
        'GMVP_OOS_RVR.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        'GMVP_OOS_RVR.pdf',
        bbox_inches='tight'
    )

    plt.show()
    plt.close(fig)
def main():
    Y, market = load_returns(INPUT_FILE)
    n, N = Y.shape #n=172, N=10, WINDOW=120
    if n <= WINDOW: #T_train=120,T_OOS=172-120=52, OOS=out of sample
        raise ValueError(f'Сарын өгөөж {n}; сургалтын цонх {WINDOW}-оос их байх ёстой.')
    print(f'Өгөөж: {n} сар | Хувьцаа: {N} | Цонх: {WINDOW} сар')
    print(f'Out-of-sample: {n - WINDOW} сар | N/T = {N/WINDOW:.4f}')

    # бүх сарын өгөөжийг ашиглан зургаан ковариацын матрицыг үнэлнэ
    full_sigmas, full_alphas = estimate_covariances(Y.to_numpy(), market.to_numpy())
    full_weights = pd.DataFrame(
        {f'GMVP_{method}': solve_long_only_gmvp(full_sigmas[method])
         for method in METHODS}, index=Y.columns
    )
    #Арга бүрийн ковариацын үнэлгээнд суурилсан in-sample портфелийн стандарт хазайлт
    full_risk = pd.DataFrame(
        {'Portfolio_Risk': {
            f'GMVP_{method}': float(np.sqrt(
                full_weights[f'GMVP_{method}'].to_numpy()
                @ full_sigmas[method]
                @ full_weights[f'GMVP_{method}'].to_numpy()
            )) for method in METHODS}}
    )
    descriptive_stats = pd.DataFrame({
        'Дундаж': Y.mean(), 'Стандарт хазайлт': Y.std(ddof=1),
        'Минимум': Y.min(), 'Максимум': Y.max(),
        'Асимметр': Y.skew(), 'Эксцесс': Y.kurt()
    })
    corr_matrix = Y.corr()
    # Хоёр дахь хэсэг: 120 сарын rolling-window, 53 OOS сар
    oos_records = []
    weight_records = []
    alpha_records = []
    # Хугацаа бүрд өмнөх 120 сарын өгөөжөөр ковариацын матриц үнэлнэ.
    # Тэр матрицаар GMVP жинг тооцно.
    #Дараагийн нэг сарын бодит өгөөжөөр портфелийн өгөөжийг үнэлнэ.
    for t in range(WINDOW, n):
        training = Y.iloc[t-WINDOW:t]
        market_training = market.iloc[t-WINDOW:t]
        actual = Y.iloc[t].to_numpy(dtype=float)
        date = Y.index[t]
        sigmas, alphas = estimate_covariances(training.to_numpy(), market_training.to_numpy())
        row = {'Date': date}
        arow = {'Date': date}
        for method in METHODS:
            w = solve_long_only_gmvp(sigmas[method])
            row[method] = float(w @ actual)
            for stock, weight in zip(STOCKS, w):
                weight_records.append({'Date': date, 'Method': method,
                                       'Stock': stock, 'Weight': float(weight)})
        for method in ['Var', 'Diag', 'Corr', 'SF']:
            arow[method] = alphas[method]
        oos_records.append(row)
        alpha_records.append(arow)
        print(f'{t-WINDOW+1:02d}/{n-WINDOW} | {date:%Y-%m} | OOS тооцоолсон')

    oos = pd.DataFrame(oos_records).set_index('Date')

    # Зургаан үнэлэгчийн OOS сарын өгөөжийн график
    plot_oos_returns(oos)

    M = len(oos)
    oos_mean = oos.mean(axis=0)
    oos_variance = oos.var(axis=0, ddof=1)
    oos_volatility = oos.std(axis=0, ddof=1)
    oos_volatility_ann = np.sqrt(12) * oos_volatility
    rvr_percent = 100 * (
        1 - oos_variance / oos_variance["Sample"]
    )
    plot_rvr(rvr_percent)
    rf_monthly = 0.0
    excess_returns = oos - rf_monthly
    sharpe_monthly = (
        excess_returns.mean(axis=0)
        / excess_returns.std(axis=0, ddof=1)
    )
    sharpe_annual = np.sqrt(12) * sharpe_monthly
    performance = pd.DataFrame({
        "Mean_Monthly_Return": oos_mean,
        "OOS_Monthly_Variance": oos_variance,
        "OOS_Monthly_Volatility": oos_volatility,
        "OOS_Annual_Volatility": oos_volatility_ann,
        "RVR_Percent": rvr_percent,
        "Sharpe_Monthly": sharpe_monthly,
        "Sharpe_Annual": sharpe_annual
    })
    performance.index.name = "Method"
    weights = pd.DataFrame(weight_records)
    alphas = pd.DataFrame(alpha_records).set_index('Date')
    # Each row = OOS holding month, columns = 10 target weights.
    weight_tables = {method: weights.loc[weights['Method'] == method]
                     .pivot(index='Date', columns='Stock', values='Weight')[STOCKS]
                     for method in METHODS}
    turnover_records = []
    hhi_records = []
    stability_records = []

    for method in METHODS:

        W = weight_tables[method]

        # 1. HHI: 52 сарын төвлөрөл
        hhi = (W ** 2).sum(axis=1)

        for date, value in hhi.items():
            hhi_records.append({
                "Date": date,
                "Method": method,
                "HHI": float(value)
            })

        # 2. Portfolio Turnover: 51 удаагийн эргэлт
        turnover_values = []

        for j in range(1, len(W)):

            w_previous = W.iloc[j - 1].to_numpy(dtype=float)
            previous_date = W.index[j - 1]

            r_previous = Y.loc[
                previous_date, STOCKS
            ].to_numpy(dtype=float)

            rp_previous = float(w_previous @ r_previous)

            w_before_rebalance = (
                w_previous * (1 + r_previous)
                / (1 + rp_previous)
            )

            w_new = W.iloc[j].to_numpy(dtype=float)

            turnover = 0.5 * np.sum(
                np.abs(w_new - w_before_rebalance)
            )

            turnover_values.append(turnover)

            turnover_records.append({
                "Date": W.index[j],
                "Method": method,
                "Turnover": float(turnover)
            })

        # 3. Арга бүрийн дундаж үзүүлэлт
        stability_records.append({
            "Method": method,
            "Mean_Turnover": float(np.mean(turnover_values)),
            "Mean_HHI": float(hhi.mean())
        })

    turnover_df = pd.DataFrame(turnover_records)
    hhi_df = pd.DataFrame(hhi_records)

    stability = pd.DataFrame(
        stability_records
    ).set_index("Method")

    print("\nPortfolio Weight Stability:")
    print(stability.round(6))
        # ==========================================
    # Statistical Comparison: HAC Test
    # ==========================================

    hac_lags = 3

    hac_records = []
    squared_difference_records = []

    for method in METHODS:

        if method == "Sample":
            continue

        # Sample аргатай харьцуулсан квадрат өгөөжийн ялгаа
        d = oos[method] ** 2 - oos["Sample"] ** 2

        for date, value in d.items():
            squared_difference_records.append({
                "Date": date,
                "Method": method,
                "Squared_Return_Difference": float(value)
            })

        # d_t = mu_d + error_t
        X_hac = np.ones((len(d), 1))

        model = sm.OLS(
            d.to_numpy(dtype=float),
            X_hac
        ).fit(
            cov_type="HAC",
            cov_kwds={
                "maxlags": hac_lags,
                "use_correction": True
            },
            use_t=False
        )

        hac_records.append({
            "Method": method,
            "Mean_Difference": float(model.params[0]),
            "HAC_SE": float(model.bse[0]),
            "HAC_t": float(model.tvalues[0]),
            "p_value": float(model.pvalues[0])
        })

    hac_results = pd.DataFrame(
    hac_records
        ).set_index("Method")

# ==========================================
# Holm Multiple Testing Correction
# ==========================================

    hac_results["p_value_Holm"] = multipletests(
        hac_results["p_value"],
        method="holm"
    )[1]

    squared_differences = pd.DataFrame(
        squared_difference_records
    )

    print("\nHAC Statistical Comparison:")
    print(hac_results.round(6))
        # ==========================================
    # N/T Sensitivity Analysis
    # Common OOS period for all windows
    # ==========================================

    sensitivity_records = []
    sensitivity_returns_records = []

    for training_window in [60, 90, 120]:

        window_oos_records = []
        window_weights = {method: [] for method in METHODS}

        # Бүх хувилбарт OOS эхлэх хугацаа ижил
        for t in range(WINDOW, n):

            training = Y.iloc[
                t - training_window:t
            ]

            market_training = market.iloc[
                t - training_window:t
            ]

            actual = Y.iloc[t].to_numpy(dtype=float)
            date = Y.index[t]

            sigmas, _ = estimate_covariances(
                training.to_numpy(),
                market_training.to_numpy()
            )

            row = {"Date": date}

            for method in METHODS:

                w = solve_long_only_gmvp(sigmas[method])

                row[method] = float(w @ actual)
                window_weights[method].append(w)

            window_oos_records.append(row)

        window_oos = pd.DataFrame(
            window_oos_records
        ).set_index("Date")

        for method in METHODS:

            returns = window_oos[method]
            sample_returns = window_oos["Sample"]

            variance = returns.var(ddof=1)
            volatility = returns.std(ddof=1)

            rvr = 100 * (
                1 - variance / sample_returns.var(ddof=1)
            )

            sharpe = (
                np.sqrt(12) * returns.mean() / volatility
            )

            W = np.asarray(window_weights[method])

            hhi_values = np.sum(W ** 2, axis=1)

            turnover_values = []

            for j in range(1, len(W)):

                w_previous = W[j - 1]

                previous_date = window_oos.index[j - 1]

                r_previous = Y.loc[
                    previous_date, STOCKS
                ].to_numpy(dtype=float)

                rp_previous = float(
                    w_previous @ r_previous
                )

                w_before_rebalance = (
                    w_previous * (1 + r_previous)
                    / (1 + rp_previous)
                )

                turnover = 0.5 * np.sum(
                    np.abs(
                        W[j] - w_before_rebalance
                    )
                )

                turnover_values.append(turnover)

            sensitivity_records.append({
                "Window": training_window,
                "N_T": N / training_window,
                "Method": method,
                "OOS_Months": len(returns),
                "Mean_Monthly_Return": returns.mean(),
                "OOS_Monthly_Variance": variance,
                "OOS_Annual_Volatility": np.sqrt(12) * volatility,
                "RVR_Percent": rvr,
                "Sharpe_Annual": sharpe,
                "Mean_Turnover": np.mean(turnover_values),
                "Mean_HHI": np.mean(hhi_values)
            })

            for date, value in returns.items():

                sensitivity_returns_records.append({
                    "Window": training_window,
                    "Date": date,
                    "Method": method,
                    "OOS_Return": float(value)
                })

    sensitivity_results = pd.DataFrame(
        sensitivity_records
    )

    sensitivity_returns = pd.DataFrame(
        sensitivity_returns_records
    )

    print("\nN/T Sensitivity Analysis:")
    print(sensitivity_results.round(6).to_string(index=False))
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # Өмнөх in-sample үр дүнг тусад нь хадгална.
        Y.to_excel(writer, sheet_name='Y')
        descriptive_stats.to_excel(writer, sheet_name='Descriptive_Statistics')
        corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        pd.DataFrame(full_sigmas['Sample'], index=Y.columns, columns=Y.columns).to_excel(writer, sheet_name='S')
        for method in ['Var', 'Diag', 'Corr', 'SF', 'NLS']:
            pd.DataFrame(full_sigmas[method], index=Y.columns, columns=Y.columns).to_excel(
                writer, sheet_name=f'Sigma_{method}')
        full_weights.to_excel(writer, sheet_name='GMVP_Weights')
        full_risk.to_excel(writer, sheet_name='GMVP_Risk')
        pd.Series(full_alphas, name='Alpha').to_frame().to_excel(writer, sheet_name='Full_Alpha')
        # Rolling-window үр дүн
        oos.to_excel(writer, sheet_name='OOS_Returns')
        alphas.to_excel(writer, sheet_name='Shrinkage_Alpha')
        performance.to_excel(writer, sheet_name='OOS_Performance')
        stability.to_excel(writer, sheet_name='Weight_Stability')
        turnover_df.to_excel(writer, sheet_name='Turnover_Monthly', index=False)
        hhi_df.to_excel(writer, sheet_name='HHI_Monthly', index=False)
        hac_results.to_excel(writer, sheet_name='HAC_Comparison')
        squared_differences.to_excel(
            writer, sheet_name='Squared_Return_Differences', index=False
        )
        sensitivity_results.to_excel(
            writer,
            sheet_name="NT_Sensitivity",
            index=False
        )

        sensitivity_returns.to_excel(
            writer,
            sheet_name="NT_OOS_Returns",
            index=False
        )
        for method, table in weight_tables.items():
            table.to_excel(writer, sheet_name=f'Weights_{method}')
        pd.DataFrame({
            'Parameter': ['Input', 'Training months', 'Number of stocks',
                          'Monthly return observations', 'OOS months', 'N/T',
                          'First OOS date', 'Last OOS date'],
            'Value': [INPUT_FILE.name, WINDOW, N, n, n-WINDOW, N/WINDOW,
                      str(oos.index[0].date()), str(oos.index[-1].date())]
        }).to_excel(writer, sheet_name='Info', index=False)
    print('\nБүх өгөгдлийн GMVP жин:')
    print(full_weights.round(5))
    print(f'\nҮр дүн хадгаллаа: {OUTPUT_FILE}')
    print('\nOut-of-sample өгөөжийн эхний 5 сар:')
    print(oos.head().round(5).to_string())
    print('\nАрга бүрийн жингийн нийлбэрийн шалгалт:')
    for method, table in weight_tables.items():
        print(f'{method}: min={table.sum(axis=1).min():.8f}, max={table.sum(axis=1).max():.8f}')

    print("\nOOS сарын тоо:", M)
    print("\nOOS Performance:")
    print(performance.round(6))

if __name__ == '__main__':
    main()
