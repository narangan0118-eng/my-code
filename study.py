import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from pypfopt.risk_models import CovarianceShrinkage

writer = pd.ExcelWriter("Ledoit_Wolf_Results.xlsx", engine="openpyxl")

close = pd.read_excel(
    "data10com.xlsx",
    sheet_name="close",
    thousands=","
)
close = close.set_index("Date")
close = close.apply(pd.to_numeric, errors="coerce")
close.round(5).to_excel(writer, sheet_name="close")

Y = close.pct_change().dropna()
Y.round(5).to_excel(writer, sheet_name="Y")

n = Y.shape[0]     
N = Y.shape[1]      
print("Ажиглалтын тоо n =", n)
print("Үнэт цаасны тоо N =", N)

S = Y.cov(ddof=0)
S_matrix = S.to_numpy()
print("Sample Covariance Matrix S:")
print(S_matrix.round(5))
S.round(5).to_excel(writer, sheet_name="S")

mu_t = np.trace(S_matrix)
print("trace(S) = ", mu_t.round(5))
mu = np.trace(S_matrix) / N
print("Mu = ", round(mu,5))
T_var = mu * pd.DataFrame(
    np.eye(N),
    index=S.index,
    columns=S.columns
)
print("Constant Variance Target Matrix T_var:")
print(T_var.round(5))
T_var.round(5).to_excel(writer, sheet_name="T_var")

#gamma = np.sum((S.values - T_var.values)**2)
#D = S.values - T_var
#gamma = np.sum(D**2)

gamma_var = np.linalg.norm(S_matrix - T_var.values, 'fro')**2
print("Gamma_var =", gamma_var.round(5))

Y_np = Y.to_numpy()
Y_tilde = Y_np - np.mean(Y_np, axis=0)

pi = 0.0
for i in range(N):
    for j in range(N):
        dev = Y_tilde[:, i] * Y_tilde[:, j] - S.iloc[i, j]
        pi_ij = np.sum(dev**2) / n
        pi +=pi_ij

print("pi = ",pi.round(5))

alpha_var = (pi / n) / gamma_var
print("alpha_var =", round(alpha_var, 5))
#lw = LedoitWolf().fit(Y_np)
#print("Sklearn Alpha =", round(lw.shrinkage_, 5))

Sigma_var = alpha_var * T_var + (1 - alpha_var) * S_matrix
print("Shrunk Covariance Matrix (Constant Variance Target) Sigma_var:")
print(Sigma_var.round(5))
Sigma_var.round(5).to_excel(writer, sheet_name="Sigma_var")

T_diag = pd.DataFrame(
    np.diag(np.diag(S_matrix)),
    index=S.index,
    columns=S.columns
)
print("Diagonal Target Matrix T_diag:")
print(T_diag.round(5))
T_diag.round(5).to_excel(writer, sheet_name="T_diag")

gamma_diag = np.linalg.norm(S_matrix - T_diag.values, 'fro')**2
print("gamma_diag =", round(gamma_diag, 5))

rho = 0.0
for i in range(N):
    dev_diag = Y_tilde[:, i]**2 - S_matrix[i, i]
    rho_i = np.sum(dev_diag**2) / n
    rho += rho_i

print("rho =", round(rho, 5))

alpha_d = (pi - rho) / (n * gamma_diag)
alpha_diag = max(0.0, min(1.0, alpha_d))
print("alpha_diag =", round(alpha_diag, 5))

Sigma_diag = alpha_diag * T_diag + (1 - alpha_diag) * S_matrix
print("Shrunk Covariance Matrix (Diagonal Target) Sigma_diag:")
print(Sigma_diag.round(5))
Sigma_diag.round(5).to_excel(writer, sheet_name="Sigma_diag")

#Table6
R = Y.corr()
print(R.round(5))
R.round(5).to_excel(writer, sheet_name="Corr")

r_bar = (R.values.sum() - N) / (N * (N - 1))
print(r_bar.round(5))

variances = np.diag(S_matrix)
std_devs = np.sqrt(variances)
table = pd.DataFrame(
    [variances, std_devs],
    index=["Дисперс", "Стандарт хазайлт"],
    columns=Y.columns
)
table.round(5).to_excel(writer, sheet_name="Var_std_dev")

T_corr = r_bar * np.outer(std_devs, std_devs)
np.fill_diagonal(T_corr, variances)
T_corr = pd.DataFrame(
    T_corr,
    index=S.index,
    columns=S.columns
)
print("Constant Correlation Target T_corr:")
print(T_corr.round(5))
T_corr.round(5).to_excel(writer, sheet_name="T_corr")

gamma_corr = np.linalg.norm(S_matrix - T_corr, 'fro')**2
print("gamma_corr =", round(gamma_corr, 5))

pi_diag = 0.0
for i in range(N):
    dev_ii = Y_tilde[:, i]**2 - S_matrix[i, i]
    pi_diag += np.sum(dev_ii**2) / n
print("pi_diag = ", pi_diag.round(5))

pi_off = 0.0
for i in range(N):
    for j in range(N):
        if i != j:
            dev_ii = Y_tilde[:, i]**2 - S_matrix[i, i]
            dev_jj = Y_tilde[:, j]**2 - S_matrix[j, j]
            cov_s_ii_jj = np.sum(dev_ii * dev_jj) / n
            
            term = cov_s_ii_jj * r_bar * (std_devs[j] / std_devs[i])
            pi_off += term
print("pi_off = ", pi_off.__round__(5))
pi_corr = pi_diag + pi_off
print("Pi_corr =", round(pi_corr, 5))

alpha_c = (pi_corr / n) / gamma_corr
alpha_corr = max(0.0, min(1.0, alpha_c))
print("alpha_corr =", round(alpha_corr, 5))

Sigma_corr = alpha_corr * T_corr + (1 - alpha_corr) * S_matrix
print("Shrunk Covariance Matrix (Constant Correlation Target) Sigma_corr:")
print(Sigma_corr.round(5))
Sigma_corr.round(5).to_excel(writer, sheet_name="Sigma_corr")


close.index = pd.to_datetime(close.index)
start_date = close.index.min().strftime("%Y-%m-%d")

sp500_data = yf.download(
    "^GSPC", start=start_date, end=None, interval="1mo", auto_adjust=True
)

if "Close" in sp500_data.columns:
    sp500 = sp500_data["Close"]
    if isinstance(sp500, pd.DataFrame):
        sp500 = sp500.squeeze()
else:
    sp500 = sp500_data.iloc[:, 0]

sp500.index = sp500.index.tz_localize(None)
sp500 = sp500[sp500.index <= close.index.max()]
print(sp500)
sp500.round(5).to_excel(writer, sheet_name="sp500")

sp500_return = sp500.pct_change().dropna()
print(sp500_return.round(5))
sp500_return.round(5).to_excel(writer, sheet_name="sp500_return")

Y.index = pd.to_datetime(Y.index).to_period("M")
sp500_return.index = pd.to_datetime(sp500_return.index).to_period("M")

combined_Y = Y.join(sp500_return.rename("Market"), how="inner").dropna()
print(combined_Y.round(5))
combined_Y.round(5).to_excel(writer, sheet_name="combined_Y")

Y_clean = combined_Y.drop(columns=["Market"]) #Y-тэй ижил гэсэн үг
R_m = combined_Y["Market"]
print(R_m)
R_m.round(5).to_excel(writer, sheet_name="R_m")

Y_np = Y_clean.to_numpy()
R_m_np = R_m.to_numpy()

var_m = np.var(R_m_np, ddof=0)
print("Var(R_m) = ",var_m.round(5))

cov_m = np.cov(Y_np.T, R_m_np, ddof=0)[:-1, -1]  # Cov(Y_i, R_m)
print("Cov(Y_i, R_m) = ",cov_m.round(5))
cov_m_df = pd.DataFrame(
    cov_m,
    index=Y_clean.columns,
    columns=["Cov(Y_i, R_m)"]
)
cov_m_df.round(5).to_excel(writer, sheet_name="Cov(Y_i,R_m)")

betas = cov_m / var_m
print("betas = ",betas.round(5))
betas_df = pd.DataFrame(
    betas,
    index=Y_clean.columns,
    columns=["betas"]
)
betas_df.round(5).to_excel(writer, sheet_name="betas")

Y_demeaned = Y_np - np.mean(Y_np, axis=0)
R_m_demeaned = R_m_np - np.mean(R_m_np)
residuals = Y_demeaned - np.outer(R_m_demeaned, betas)
print(residuals)
var_e = np.var(residuals, axis=0, ddof=0)  # Residual variances (N x 1)
print("Var(epsilon_i) = ",var_e.round(5))
var_e_df = pd.DataFrame(
    var_e,
    index=Y_clean.columns,
    columns=["Var(epsilon_i)"]
)
var_e_df.round(5).to_excel(writer, sheet_name="Var(epsilon_i)")

T_SF_matrix = var_m * np.outer(betas, betas)
np.fill_diagonal(T_SF_matrix, (betas**2) * var_m + var_e)

T_SF = pd.DataFrame(T_SF_matrix, index=S.index, columns=S.columns)
print("Нэг хүчин зүйлийн зорилтот матриц T_SF:")
print(T_SF.round(5))
T_SF.round(5).to_excel(writer, sheet_name="T_SF")

gamma_SF = np.linalg.norm(S_matrix - T_SF, 'fro')**2
print("gamma_SF =", round(gamma_SF, 5))

pi_factor = 0.0
for i in range(N):
    for j in range(N):
        dev = Y_demeaned[:, i] * Y_demeaned[:, j] - S_matrix[i, j]
        pi_ij = np.sum(dev**2) / n
        pi_factor += pi_ij

print("pi_factor =", round(pi_factor, 5))

alpha_SF_s = (pi_factor / n) / gamma_SF
alpha_SF= max(0.0, min(1.0, alpha_SF_s))

print("alpha_SF =", round(alpha_SF, 5))

Sigma_SF_matrix = alpha_SF * T_SF + (1 - alpha_SF) * S_matrix
Sigma_SF = pd.DataFrame(Sigma_SF_matrix, index=S.index, columns=S.columns)
print("Shrunk Covariance Matrix (Single-Index Target) Sigma_SF:")
print(Sigma_SF.round(5))
Sigma_SF.round(5).to_excel(writer, sheet_name="Sigma_SF")

Y_nls = Y_clean.to_numpy()
Y_centered = Y_nls - np.mean(Y_nls, axis=0)

eigenvalues_sample, U_nls = np.linalg.eigh(S)
print(eigenvalues_sample.round(5))
print(U_nls.round(5))

sort_idx = np.argsort(eigenvalues_sample)
print(sort_idx.round(5))
lambda_sample = eigenvalues_sample[sort_idx]
lambda_sample_df = pd.DataFrame(
    lambda_sample,
    index=Y_clean.columns,
    columns=["lambda_sample"]
)
print(lambda_sample_df.round(5))

U_nls = U_nls[:, sort_idx]

c = N / n
print("c = ", round(c, 5))
h = N ** (-1 / 3)  # Kernel bandwidth parameter
print("h = ", round(h, 5))

lambda_safe = np.maximum(lambda_sample, 1e-8)
log_lambda = np.log(lambda_safe)

z_mat = (log_lambda[:, None] - log_lambda[None, :]) / h
print(z_mat.round(5))
abs_z = np.abs(z_mat)
sqrt_5 = np.sqrt(5)

k_x = np.where(abs_z <= sqrt_5, 3 / (4 * sqrt_5) * (1 - z_mat**2 / 5), 0)
print(k_x.round(5))
k_x_df = pd.DataFrame(
    k_x,
    index=Y_clean.columns,
    columns=Y_clean.columns
)
k_x_df.round(5).to_excel(writer, sheet_name="k(x)")

e_x = np.where(
    abs_z <= sqrt_5,
    3 / (10 * np.pi) * z_mat
    + 3
    / (4 * sqrt_5 * np.pi)
    * (1 - z_mat**2 / 5)
    * np.log(np.abs((sqrt_5 - z_mat) / (sqrt_5 + z_mat + 1e-12))),
    0,
)
print(e_x.round(5))
e_x_df = pd.DataFrame(
    e_x,
    index=Y_clean.columns,
    columns=Y_clean.columns
)
e_x_df.round(5).to_excel(writer, sheet_name="e(x)")

f_hat = np.sum(k_x, axis=1) / (N * h * lambda_safe)
print(f_hat)
m_hat = np.sum(e_x, axis=1) / (N * h * lambda_safe)
print(m_hat)
df_nls = pd.DataFrame(
    [f_hat, m_hat],
    index=["f_hat", "m_hat"],
    columns=Y_clean.columns
)

df_nls.to_excel(
    writer,
    sheet_name="NLS",
    index=True
)

d_huvaari = (1 - c - c * lambda_safe * m_hat) ** 2 + (c * np.pi * lambda_safe * f_hat) ** 2
d_hat = lambda_safe / d_huvaari
print(d_hat)
d_hat_df = pd.DataFrame(
    d_hat,
    index=Y_clean.columns,
    columns=["d_hat"]
)
d_hat_df.round(5).to_excel(writer, sheet_name="d_hat")

for i in range(1, N):
    if d_hat[i] < d_hat[i - 1]:
        d_hat[i] = d_hat[i - 1]

Sigma_NLS_matrix = U_nls @ np.diag(d_hat) @ U_nls.T
Sigma_NLS = pd.DataFrame(
    Sigma_NLS_matrix, index=S.index, columns=S.columns
)

print(
    "\nShrunk Covariance Matrix (Analytical Nonlinear Shrinkage) Sigma_NLS:"
)
print(Sigma_NLS.round(5))
Sigma_NLS.round(5).to_excel(writer, sheet_name="Sigma_NLS")

eigen_comparison = pd.DataFrame(
    {
        "Түүврийн хувийн утга (Sample)": lambda_sample,
        "Агшсан хувийн утга (NLS)": d_hat,
    },
    index=[f"lambda_{i+1}" for i in range(N)],
)

print("\nХувийн утгуудын харьцуулалт:")
print(eigen_comparison.round(5))
eigen_comparison.round(5).to_excel(writer, sheet_name="Eigenvalues_NLS")

writer.close()


