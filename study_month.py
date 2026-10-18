import pandas as pd
import numpy as np

writer = pd.ExcelWriter("Ledoit_Wolf_Results_monthly.xlsx", engine="openpyxl")

close = pd.read_excel("monthly.xlsx", sheet_name="close", thousands=",")
close = close.set_index("Date")
close = close.apply(pd.to_numeric, errors="coerce")
close.round(5).to_excel(writer, sheet_name="close")

Y = close.drop(columns=["SP500"]).pct_change().dropna()
Y.round(5).to_excel(writer, sheet_name="Y")

n = Y.shape[0]     
N = Y.shape[1]      
print("Ажиглалтын тоо n =", n)
print("Үнэт цаасны тоо N =", N)

descriptive_stats = pd.DataFrame({
    "Дундаж": Y.mean(),
    "Стандарт хазайлт": Y.std(ddof=1),
    "Минимум": Y.min(),
    "Максимум": Y.max(),
    "Асимметр": Y.skew(),
    "Эксцесс": Y.kurt()
})
descriptive_stats.index.name = "Хөрөнгө"
descriptive_stats.round(4).to_excel(writer, sheet_name="Descriptive_Statistics")

#Түүврийн ковариацын матриц (Sample covariance matrix) ---
S = Y.cov(ddof=0)
S_matrix = S.to_numpy()
S.round(5).to_excel(writer, sheet_name="S")

#Тогтмол вариацын зорилтот матриц (Constant Variance Target) ---
mu_hat = np.trace(S.to_numpy()) / N
print("mu_hat = ", round(mu_hat,5))

T_var = pd.DataFrame(
    mu_hat * np.eye(N),
    index=Y.columns,
    columns=Y.columns
)
T_var.round(5).to_excel(writer, sheet_name="T_var")

Y_tilde = Y - Y.mean()
X = Y_tilde.to_numpy()
S_matrix = S.to_numpy()
b_sum1 = 0.0
for t in range(n):
    Z_t = np.outer(X[t], X[t])
    b_sum1 += np.sum((Z_t - S_matrix)**2)
b_hat = b_sum1 / (n**2 * N)
d_hat_var = (
    np.sum((S_matrix - T_var.to_numpy())**2) / N
)
if d_hat_var > 0:
    alpha_hat_var = np.clip(b_hat / d_hat_var, 0, 1)
else:
    alpha_hat_var = 1.0
print(alpha_hat_var.round(5))

Sigma_var = alpha_hat_var * T_var + (1 - alpha_hat_var) * S_matrix
Sigma_var.round(5).to_excel(writer, sheet_name="Sigma_var")

#Диагональ зорилтот матриц (Diagonal Target) ---
T_diag = pd.DataFrame(
    np.diag(np.diag(S_matrix)),
    index=S.index,
    columns=S.columns
)
T_diag.round(5).to_excel(writer, sheet_name="T_diag")

b_sum2 = 0.0
for t in range(n):
    Z_t = np.outer(X[t], X[t])
    difference = Z_t - S_matrix
    # Диагональ элементүүдийг хасах
    np.fill_diagonal(difference, 0)
    b_sum2 += np.sum(difference**2)
b_hat_diag = b_sum2 / (n**2 * N)
print("b_hat_diag =", b_hat_diag.round(5))
d_hat_diag = (
    np.sum((S_matrix - T_diag.to_numpy())**2) / N
)
print("d_hat_diag =", d_hat_diag.round(5))
if d_hat_diag > 0:
    alpha_hat_diag = np.clip(
        b_hat_diag / d_hat_diag, 0, 1
    )
else:
    alpha_hat_diag = 1.0
print("alpha_hat_diag =", alpha_hat_diag.round(5))
Sigma_diag = ((1 - alpha_hat_diag) * S + alpha_hat_diag * T_diag)
Sigma_diag.round(5).to_excel(writer, sheet_name="Sigma_diag")

#Тогтмол корреляцын зорилтот матриц (Constant Correlation Target) ---
std_dev = np.sqrt(np.diag(S_matrix))
outer_std = np.outer(std_dev, std_dev)
R_matrix = S_matrix / outer_std
R = pd.DataFrame(R_matrix, index=S.index, columns=S.columns)
R.round(5).to_excel(writer, sheet_name="R")

sum_r = np.sum(R_matrix) - np.sum(np.diag(R_matrix))  # Диагональ бус элементүүдийн нийлбэр (i != j)
r_bar = sum_r / (N * (N - 1))
print("r_bar (Корреляцын дундаж) =", round(r_bar, 5))

T_corr_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            T_corr_matrix[i, j] = S_matrix[i, i]
        else:
            T_corr_matrix[i, j] = r_bar * std_dev[i] * std_dev[j]

T_corr = pd.DataFrame(T_corr_matrix, index=S.index, columns=S.columns)
T_corr.round(5).to_excel(writer, sheet_name="T_corr")

b_sum_corr = 0.0
for t in range(n):
    Z_t = np.outer(X[t], X[t])
    b_sum_corr += np.sum((Z_t - S_matrix) ** 2)
b_hat_corr = b_sum_corr / (n**2 * N)
print("b_hat_corr =", round(b_hat_corr, 5))

diff_corr = S_matrix - T_corr_matrix
d_hat_corr = np.sum(diff_corr**2) / N
print("d_hat_corr =", round(d_hat_corr, 5))

var_y2 = np.zeros(N)
for i in range(N):
    var_y2[i] = np.mean((X[:, i] ** 2 - S_matrix[i, i]) ** 2)
sum_var_y2 = np.sum(var_y2)

theta_sum = 0.0
for i in range(N):
    for j in range(N):
        if i != j:
            # theta_ii_ij
            theta_ii_ij = np.mean(
                (X[:, i] ** 2 - S_matrix[i, i])
                * (X[:, i] * X[:, j] - S_matrix[i, j])
            )
            # theta_jj_ij
            theta_jj_ij = np.mean(
                (X[:, j] ** 2 - S_matrix[j, j])
                * (X[:, i] * X[:, j] - S_matrix[i, j])
            )

            term = (np.sqrt(S_matrix[j, j] / S_matrix[i, i]) * theta_ii_ij) + (
                np.sqrt(S_matrix[i, i] / S_matrix[j, j]) * theta_jj_ij
            )
            theta_sum += term

c_hat_corr = (1.0 / (n * N)) * (sum_var_y2 + (r_bar / 2.0) * theta_sum)
print("c_hat_corr =", round(c_hat_corr, 5))

if d_hat_corr > 0:
    alpha_hat_corr = np.clip((b_hat_corr - c_hat_corr) / d_hat_corr, 0, 1)
else:
    alpha_hat_corr = 1.0
print("alpha_hat_corr =", round(alpha_hat_corr, 5))

Sigma_corr = (1 - alpha_hat_corr) * S + alpha_hat_corr * T_corr
Sigma_corr.round(5).to_excel(writer, sheet_name="Sigma_corr")

#Нэг хүчин зүйлийн зорилтот матриц (Single-Index Target Matrix & Shrinkage) ---
R_m = close["SP500"].pct_change().loc[Y.index].to_numpy()
R_m_bar = np.mean(R_m)
sigma2_m = np.mean((R_m - R_m_bar) ** 2)

Y_bar = np.mean(X, axis=0)
R_m_tilde = R_m - R_m_bar
dem_beta = np.sum(R_m_tilde ** 2)

beta_hat = np.sum((X - Y_bar) * R_m_tilde[:, np.newaxis], axis=0) / dem_beta

epsilon = (X - Y_bar) - np.outer(R_m_tilde, beta_hat)
sigma2_eps = np.mean(epsilon ** 2, axis=0)

T_SF_matrix = np.outer(beta_hat, beta_hat) * sigma2_m + np.diag(sigma2_eps)
T_SF = pd.DataFrame(T_SF_matrix, index=S.index, columns=S.columns)
T_SF.round(5).to_excel(writer, sheet_name="T_SF")

d_hat_SF = np.sum((S_matrix - T_SF_matrix) ** 2) / N
print("d_hat_SF =", round(d_hat_SF, 5))

Z_centered = X - Y_bar
b_sum_SF = 0.0
for t in range(n):
    Z_t = np.outer(Z_centered[t], Z_centered[t])
    b_sum_SF += np.sum((Z_t - S_matrix) ** 2)
b_hat_SF = b_sum_SF / (n**2 * N)
print("b_hat_SF =", round(b_hat_SF, 5))

q_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        term2 = Z_centered[:, i] * Z_centered[:, j] - S_matrix[i, j]
        if i == j:
            term1 = Z_centered[:, i] ** 2 - S_matrix[i, i]
        else:
            term1 = (
                beta_hat[j] * Z_centered[:, i] * R_m_tilde
                + beta_hat[i] * Z_centered[:, j] * R_m_tilde
                - beta_hat[i] * beta_hat[j] * (R_m_tilde ** 2)
                - T_SF_matrix[i, j]
            )
        q_matrix[i, j] = np.sum(term1 * term2) / n

c_hat_SF = np.sum(q_matrix) / (n * N)
print("c_hat_SF =", round(c_hat_SF, 5))

if d_hat_SF > 0:
    alpha_hat_SF = np.clip((b_hat_SF - c_hat_SF) / d_hat_SF, 0, 1)
else:
    alpha_hat_SF = 1.0
print("alpha_hat_SF =", round(alpha_hat_SF, 5))

Sigma_SF = (1 - alpha_hat_SF) * S + alpha_hat_SF * T_SF
Sigma_SF.round(5).to_excel(writer, sheet_name="Sigma_SF")

#Аналитик шугаман бус агшилтын үнэлгээ (Analytical Nonlinear Shrinkage - NLS) ---

eigenvalues, U = np.linalg.eigh(S_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
U = U[:, sorted_indices]

eigenvalues = np.maximum(eigenvalues, 1e-8)

c = N / n

h = n ** (-1/3)
h_j = eigenvalues * h  

def epanechnikov_kernel(x):
    res = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= np.sqrt(5.0)
    res[mask] = (3.0 / (4.0 * np.sqrt(5.0))) * (1.0 - x[mask] ** 2 / 5.0)
    return res

def hilbert_epanechnikov_kernel(x):
    res = np.zeros_like(x)
    mask1 = (np.abs(x) != np.sqrt(5)) & (np.abs(x) <= 1e5)
    x_m = x[mask1]
    log_term = np.log(np.abs((np.sqrt(5) - x_m) / (np.sqrt(5) + x_m)))
    res[mask1] = (-3 * x_m / (10 * np.pi)) + (3 / (4 * np.sqrt(5) * np.pi)) * (1 - (x_m ** 2) / 5) * log_term
    
    mask2 = (np.abs(x) == np.sqrt(5))
    res[mask2] = -3 * x[mask2] / (10 * np.pi)
    return res

f_hat = np.zeros(N)
Hf_hat = np.zeros(N)

for i in range(N):
    lam_i = eigenvalues[i]
    x_arg = (lam_i - eigenvalues) / h_j
    
    f_hat[i] = (1 / N) * np.sum((1 / h_j) * epanechnikov_kernel(x_arg))
    Hf_hat[i] = (1 / N) * np.sum((1 / h_j) * hilbert_epanechnikov_kernel(x_arg))

d_hat = np.zeros(N)
for i in range(N):
    lam_i = eigenvalues[i]
    term1 = 1 - c - c * np.pi * lam_i * Hf_hat[i]
    term2 = c * np.pi * lam_i * f_hat[i]
    d_hat[i] = lam_i / ((term1 ** 2) + (term2 ** 2))

D_hat = np.diag(d_hat)
Sigma_NLS_matrix = U @ D_hat @ U.T

Sigma_NLS = pd.DataFrame(Sigma_NLS_matrix, index=S.index, columns=S.columns)
Sigma_NLS.round(5).to_excel(writer, sheet_name="Sigma_NLS")


from scipy.optimize import minimize


covariance_matrices = {
    "GMVP_Sample": S_matrix,
    "GMVP_Var": Sigma_var.to_numpy(),
    "GMVP_Diag": Sigma_diag.to_numpy(),
    "GMVP_Corr": Sigma_corr.to_numpy(),
    "GMVP_SF": Sigma_SF.to_numpy(),
    "GMVP_NLS": Sigma_NLS_matrix
}

from scipy.optimize import minimize

def solve_long_only_gmvp(Sigma):

    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = (Sigma + Sigma.T) / 2

    def objective(x):
        return x @ Sigma @ x

    def gradient(x):
        return 2 * Sigma @ x

    constraints = {
        "type": "eq",
        "fun": lambda x: np.sum(x) - 1.0,
        "jac": lambda x: np.ones(N)
    }

    bounds = [(0.0, 1.0)] * N
    x0 = np.ones(N) / N

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={
            "ftol": 1e-15,
            "maxiter": 10000
        }
    )

    if not result.success:
        raise RuntimeError(result.message)

    return result.x

gmvp_weights_df = pd.DataFrame(index=Y.columns)

for name, Sigma in covariance_matrices.items():
    weights = solve_long_only_gmvp(Sigma)
    gmvp_weights_df[name] = weights

print("\n--- Long-only GMVP жингийн нийлбэр ---")
print(gmvp_weights_df.sum())

gmvp_weights_df.round(5).to_excel(writer, sheet_name="GMVP_Weights")

gmvp_risk = pd.Series(index=covariance_matrices.keys(), dtype=float)
for name, Sigma in covariance_matrices.items():
    w = gmvp_weights_df[name].to_numpy()
    portfolio_variance = w.T @ Sigma @ w
    gmvp_risk[name] = np.sqrt(portfolio_variance)

print("\n--- Long-only GMVP Багцын Эрсдэл (Standard Deviation) ---")
print(gmvp_risk.round(5))

gmvp_risk.to_frame(name="Portfolio_Risk").round(5).to_excel(writer, sheet_name="GMVP_Risk")



writer.close()
