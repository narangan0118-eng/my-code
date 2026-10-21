from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
INPUT = BASE / 'Robustness_150_Results/OOS_Returns_150_Portfolios.csv'
OUTPUT = BASE / 'Tier_RVR_Bootstrap_Comparison.xlsx'
METHODS = ['Var', 'Diag', 'Corr', 'SF', 'NLS']
PAIRS = [('High', 'Medium'), ('High', 'Low'), ('Medium', 'Low')]
B, BLOCK, SEED = 3000, 6, 20261021

df = pd.read_csv(INPUT)
df['Date'] = pd.to_datetime(df['Date'])
dates = pd.DatetimeIndex(sorted(df['Date'].unique()))
ids = df[['Portfolio_ID','Tier']].drop_duplicates().sort_values(['Tier','Portfolio_ID'])
assert len(dates) == 52 and len(ids) == 150
assert ids.groupby('Tier').size().to_dict() == {'High':50,'Low':50,'Medium':50}
cols = ['Sample'] + METHODS
arr = np.stack([df.loc[df.Portfolio_ID.eq(pid)].sort_values('Date')[cols].to_numpy(dtype=float)
                for pid in ids.Portfolio_ID])
assert arr.shape == (150, 52, 6) and np.isfinite(arr).all()
assert all((df.loc[df.Portfolio_ID.eq(pid)].sort_values('Date')['Date'].to_numpy() == dates.to_numpy()).all()
         for pid in ids.Portfolio_ID)
tiers = ids.Tier.to_numpy()

def mean_rvr(x):
    # x shape (portfolio, month, estimator)
    variances = np.var(x, axis=1, ddof=1)
    rvr = 100 * (1 - variances[:, 1:] / variances[:, [0]])
    return {g: rvr[tiers == g].mean(axis=0) for g in ['High','Medium','Low']}

observed = mean_rvr(arr)
labels = [(a,b,m,j) for a,b in PAIRS for j,m in enumerate(METHODS)]
obs = np.array([observed[a][j]-observed[b][j] for a,b,m,j in labels])
rng = np.random.default_rng(SEED)
boot = np.empty((B, len(labels)))
for b in range(B):
    starts = rng.integers(0, 52, size=(52 + BLOCK - 1)//BLOCK)
    idx = np.concatenate([(s + np.arange(BLOCK)) % 52 for s in starts])[:52]
    means = mean_rvr(arr[:, idx, :])
    boot[b] = [means[a][j]-means[c][j] for a,c,m,j in labels]

ci = np.quantile(boot, [0.025,0.975], axis=0)
# Bootstrap p-value under the null, using the centered bootstrap distribution.
centered = boot - obs
p = np.array([(1 + np.sum(np.abs(centered[:,j]) >= abs(obs[j]))) / (B+1)
              for j in range(len(labels))])
# Holm familywise correction across all 15 comparisons.
order = np.argsort(p)
adj_sorted = np.maximum.accumulate((len(p)-np.arange(len(p))) * p[order])
adj = np.empty_like(p)
adj[order] = np.minimum(1, adj_sorted)
rows=[]
for j,(a,b,m,_) in enumerate(labels):
    rows.append({'Comparison':f'{a}-{b}','Method':m,
                 'Difference_pp':obs[j], 'CI_95_Lower':ci[0,j], 'CI_95_Upper':ci[1,j],
                 'Bootstrap_p':p[j], 'Holm_p_15':adj[j],
                 'CI_excludes_zero':bool(ci[0,j]>0 or ci[1,j]<0)})
res=pd.DataFrame(rows)
with pd.ExcelWriter(OUTPUT,engine='openpyxl') as writer:
    res.to_excel(writer,index=False,sheet_name='Tier_Comparisons')
    pd.DataFrame([{'Tier':g, **dict(zip(METHODS,observed[g]))} for g in ['High','Medium','Low']]).to_excel(writer,index=False,sheet_name='Mean_RVR')
print(res.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('Saved:',OUTPUT)
