
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from pypfopt.risk_models import CovarianceShrinkage

close = pd.read_excel(
    "data10com.xlsx",
    sheet_name="close",
    thousands=","
)
close = close.set_index("Date")
close = close.apply(pd.to_numeric, errors="coerce")
Y = close.pct_change().dropna()

#Table1
n = Y.shape[0]      # Ажиглалтын тоо
N = Y.shape[1]      # Үнэт цаасны тоо
print("Ажиглалтын тоо n =", n)
print("Үнэт цаасны тоо N =", N)

S = Y.cov(ddof=0)
S_matrix = S.to_numpy()
print("Sample Covariance Matrix S:")
print(S_matrix.round(5))

#Table2
mu = np.trace(S_matrix) / N
print("\nMu = ", round(mu,5))
T1 = mu * pd.DataFrame(
    np.eye(N),
    index=S.index,
    columns=S.columns
)
print("\nConstant Variance Target Matrix T1:")
print(T1.round(5))

#Table3
#gamma = np.sum((S.values - T1.values)**2)

#D = S.values - T1
#gamma = np.sum(D**2)

gamma = np.linalg.norm(S_matrix - T1.values, 'fro')**2
print("Gamma =", gamma.round(5))

Y_np = Y.to_numpy()

Y_tilde = Y_np - np.mean(Y_np, axis=0)

pi_total = 0.0

for i in range(N):
    for j in range(N):
        dev = Y_tilde[:, i] * Y_tilde[:, j] - S.iloc[i, j]
        pi_ij = np.sum(dev**2) / n
        pi_total +=pi_ij

print(pi_total.round(5))

alpha = (pi_total / n) / gamma
print("\nAlpha =", round(alpha, 5))
#lw = LedoitWolf().fit(Y_np)
#print("Sklearn Alpha =", round(lw.shrinkage_, 5))

T3 = alpha * T1 + (1 - alpha) * S_matrix
print(T3.round(5))

#Table 4
T4 = pd.DataFrame(
    np.diag(np.diag(S_matrix)),
    index=S.index,
    columns=S.columns
)
print("\nDiagonal Target Matrix T4:")
print(T4.round(5))
T4.to_excel("T4.xlsx", index=True)

#Table5
gamma2 = np.linalg.norm(S_matrix - T4.values, 'fro')**2
print("\nGamma2 =", round(gamma2, 5))

rho = 0.0
for i in range(N):
    dev_diag = Y_tilde[:, i]**2 - S_matrix[i, i]
    rho_i = np.sum(dev_diag**2) / n
    rho += rho_i

print("Rho =", round(rho, 6))

delta_raw = (pi_total - rho) / (n * gamma2)
delta = max(0.0, min(1.0, delta_raw))
print("\nDelta (Diagonal Target) =", round(delta, 5))
print(delta)

T5 = delta * T4 + (1 - delta) * S_matrix
print(T5.round(5))
T5.to_excel("T5.xlsx", index=True)

#Table6

R = Y.corr()
print(R.round(5))
R.to_excel("Corr.xlsx", index=True)
rho_bar = (R.values.sum() - N) / (N * (N - 1))
print(rho_bar*45.round(5))
