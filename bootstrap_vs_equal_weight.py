from pathlib import Path
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import pandas as pd

BASE=Path(__file__).resolve().parent
METHODS=['Sample','Var','Diag','Corr','SF','NLS']
B=3000
BLOCK=6
SEED=20261021
rng=np.random.default_rng(SEED)
rows=[]
for tier in ['High','Medium','Low']:
    src = BASE / 'GMVP_Three_Portfolios_Results' / tier / f'monthly_rolling_window_{tier}.xlsx'
    if not src.exists():
        src=BASE/f'monthly_rolling_window_{tier}.xlsx'
    if not src.exists():
        raise FileNotFoundError(f'Excel файл олдсонгүй: {src}')
    data=pd.read_excel(src,sheet_name='Practical_OOS_Returns')
    required=METHODS+['EW_1N']
    if len(data)!=52 or not all(name in data.columns for name in required):
        raise ValueError(f'52 сарын өгөөж эсвэл багана дутуу: {src}; columns={list(data.columns)}')
    mat=data[required].to_numpy(dtype=float)
    if not np.isfinite(mat).all():raise ValueError(f'Non-finite values: {tier}')
    n=len(mat)
    # Shared month resampling within a tier; preserve pairing between strategies.
    nblocks=int(np.ceil(n/BLOCK))
    starts=rng.integers(0,n,size=(B,nblocks))
    offsets=np.arange(BLOCK)
    ix=((starts[:,:,None]+offsets[None,None,:])%n).reshape(B,-1)[:,:n]
    draws=mat[ix]
    bootvar=draws.var(axis=1,ddof=1)
    observed=mat.var(axis=0,ddof=1)
    for j,method in enumerate(METHODS):
        estimate=100*(1-observed[j]/observed[-1])
        bstat=100*(1-bootvar[:,j]/bootvar[:,-1])
        ci=np.percentile(bstat,[2.5,97.5])
        # Centered bootstrap null approximation (two-sided), not a test inversion.
        p=(1+np.sum(np.abs(bstat-estimate)>=abs(estimate)))/(B+1)
        rows.append([tier,method,float(estimate),float(ci[0]),float(ci[1]),float(p),None,n,BLOCK,B])
    print(tier, [(r[1],round(r[2],2),round(r[3],2),round(r[4],2)) for r in rows[-6:]])
for tier in ['High','Medium','Low']:
    inds=[i for i,r in enumerate(rows) if r[0]==tier]
    corrected=multipletests([rows[i][5] for i in inds],method='holm')[1]
    for i,p in zip(inds,corrected):rows[i][6]=float(p)

cols=['Portfolio','Method','RVR_vs_1N_Percent','CI_2.5_Percent','CI_97.5_Percent',
      'Bootstrap_p','Holm_p_6','OOS_Months','Block_Months','Bootstrap_B']
results=pd.DataFrame(rows,columns=cols)
notes=[['Item','Explanation'],
       ['Statistic','100*(1-var(GMVP)/var(1/N)), sample variance ddof=1'],
       ['Bootstrap','Circular moving-block, paired monthly resampling across all 7 strategies'],
       ['Block length',BLOCK],['Replicates',B],['Random seed',SEED],
       ['CI','Unadjusted percentile 95% CI for each RVR'],
       ['p-value','Two-sided centered-bootstrap approximation for RVR=0'],
       ['Holm','6 comparisons within each portfolio'],
       ['Scope','52 monthly OOS observations; no cross-portfolio familywise control'],
       ['Caution','Percentile CIs and centered p-values need not yield identical decisions']]
path=BASE/'Bootstrap_GMVP_vs_1N.xlsx'
with pd.ExcelWriter(path,engine='openpyxl') as writer:
    results.to_excel(writer,sheet_name='Bootstrap_vs_1N',index=False)
    pd.DataFrame(notes[1:],columns=notes[0]).to_excel(writer,sheet_name='Methodology',index=False)
    for sheet_name in ('Bootstrap_vs_1N','Methodology'):
        ws=writer.sheets[sheet_name]
        ws.freeze_panes='A2'
        for cell in ws[1]:
            cell.font=__import__('openpyxl').styles.Font(bold=True,color='FFFFFF')
            cell.fill=__import__('openpyxl').styles.PatternFill('solid',fgColor='17365D')
        for col in ws.columns:
            letter=col[0].column_letter
            ws.column_dimensions[letter].width=min(62,max(15,max(len(str(c.value or '')) for c in col)+2))
print('Saved',path)
