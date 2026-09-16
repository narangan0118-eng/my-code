import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf

close = pd.read_excel(
    r"C:\Users\User\Documents\My code\data10com.xlsx",
    sheet_name="close",
    thousands=","
)
close = close.set_index("Date")
close = close.apply(pd.to_numeric, errors="coerce")
returns = close.pct_change().dropna()

#Table1
n = returns.shape[0]      # Ажиглалтын тоо
N = returns.shape[1]      # Үнэт цаасны тоо
print("Ажиглалтын тоо n =", n)
print("Үнэт цаасны тоо N =", N)
S = returns.cov(ddof=0)
print("Sample Covariance Matrix S:")
print(S.round(5))

#Table2
mu = np.trace(S.values) / N
print(mu.round(5))
T1 = mu * pd.DataFrame(
    __import__("numpy").eye(N),
    index=S.index,
    columns=S.columns
)
print("\nConstant Variance Target Matrix T1:")
print(T1.round(5))

#Table3
#gamma = np.sum((S.values - T1.values)**2)

#D = S.values - T1
#gamma = np.sum(D**2)

gamma = np.linalg.norm(S.values - T1.values, 'fro')**2
print("Gamma =", gamma.round(5))

S_matrix = S.values

pi = 0

for t in range(n):
    x_t = X.iloc[t].values.reshape(-1, 1)
    outer_product = x_t @ x_t.T
    pi += np.sum((outer_product - S_matrix)**2)

pi = pi / n

print("Pi =", pi)
alpha = (pi / n) / gamma
print(alpha)
#lw = LedoitWolf().fit(returns)
#alpha = lw.shrinkage_
print("Alpha =", alpha.round(5))

T3 = alpha * T1 + (1 - alpha) * S_matrix
print("T3 = ",T3.round(5))

#Table4
