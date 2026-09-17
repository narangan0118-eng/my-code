import pandas as pd
import numpy as np
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

gamma_var = np.linalg.norm(S_matrix - T2.values, 'fro')**2
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

Sigma1 = alpha_var * T_var + (1 - alpha_var) * S_matrix
print("Shrunk Covariance Matrix (Constant Variance Target) Sigma1:")
print(Sigma1.round(5))
Sigma1.round(5).to_excel(writer, sheet_name="Sigma1")

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

Sigma2 = alpha_diag * T_diag + (1 - alpha_diag) * S_matrix
print("Shrunk Covariance Matrix (Diagonal Target) Sigma2:")
print(Sigma2.round(5))
Sigma2.round(5).to_excel(writer, sheet_name="Sigma2")

#Table6
R = Y.corr()
print(R.round(5))
R.round(5).to_excel(writer, sheet_name="Corr")

r_bar = (R.values.sum() - N) / (N * (N - 1))
print(r_bar.round(5))

variances = np.diag(S_matrix)
std_devs = np.sqrt(variances)
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

Sigma3 = alpha_corr * T_corr + (1 - alpha_corr) * S_matrix
print("Shrunk Covariance Matrix (Constant Correlation Target) Sigma3:")
print(Sigma3.round(5))
Sigma3.round(5).to_excel(writer, sheet_name="Sigma3")

writer.close()
