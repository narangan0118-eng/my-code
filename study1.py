import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
import os
import sys
from sklearn.covariance import LedoitWolf

# 1) Close унших
close = pd.read_excel(
    "data10com.xlsx",
    sheet_name="close",
    thousands=",",
    skiprows=0
)

# 2) Хэрвээ row-нууд компани, багана нь Date бол transpose хийнэ
close = close.T

# 3) Эхний мөрийг баганын нэр болгоно
close.columns = close.iloc[0]
close = close.drop(close.index[0])

# 4) Бүх утгыг numeric болгоно
close = close.apply(pd.to_numeric, errors='coerce')

# 5) Log return тооцоолох
growth = np.log(close / close.shift(1)).dropna()

# 6) Ledoit-Wolf Shrinkage
growth.columns = growth.columns.astype(str)
lw = LedoitWolf().fit(growth)
shrunk_cov = lw.covariance_

print("\n✅ Shrunk covariance matrix:")
print(shrunk_cov)

print("\n✅ Lambda (shrinkage intensity):", lw.shrinkage_)
