import numpy as np
import pandas as pd
from scipy import stats

file_path = "data10com.xlsx"
sheet_name = "cost"

norm_results = []

for column in df.columns:
    data = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(data) >= 3:
        sw_stat, sw_p = stats.shapiro(data)
        mean, std = data.mean(), data.std()
        ks_stat, ks_p = stats.kstest(data, stats.norm(loc=mean, scale=std).cdf)

        status = "Хэвийн тархалттай" if ks_p > 0.05 else "Хэвийн тархалтгүй"

        norm_results.append(
            {
                "Компани": column,
                "n": len(data),
                "Shapiro W": round(sw_stat, 4),
                "Shapiro p-val": round(sw_p, 6),
                "KS Stat": round(ks_stat, 4),
                "KS p-val": round(ks_p, 6),
                "Хэвийн тархалт Дүгнэлт": status,
            }
        )

norm_df = pd.DataFrame(norm_results)


lognorm_results = []

for column in df.columns:
    data = pd.to_numeric(df[column], errors="coerce").dropna()
    data_pos = data[data > 0]

    if len(data_pos) >= 3:
        log_data = np.log(data_pos)
        sw_stat, sw_p = stats.shapiro(log_data)

        shape, loc, scale = stats.lognorm.fit(data_pos, floc=0)
        ks_stat, ks_p = stats.kstest(
            data_pos, stats.lognorm(s=shape, loc=loc, scale=scale).cdf
        )

        status = (
            "Лог-хэвийн тархалттай"
            if ks_p > 0.05
            else "Лог-хэвийн тархалтгүй"
        )

        lognorm_results.append(
            {
                "Компани": column,
                "n": len(data_pos),
                "Log-Shapiro W": round(sw_stat, 4),
                "Log-Shapiro p-val": round(sw_p, 6),
                "Log-KS Stat": round(ks_stat, 4),
                "Log-KS p-val": round(ks_p, 6),
                "Лог-хэвийн тархалт Дүгнэлт": status,
            }
        )

lognorm_df = pd.DataFrame(lognorm_results)


merged_df = pd.merge(
    norm_df[
        ["Компани", "n", "KS p-val", "Хэвийн тархалт Дүгнэлт"]
    ].rename(columns={"KS p-val": "Normal KS p-val"}),
    lognorm_df[
        ["Компани", "Log-KS p-val", "Лог-хэвийн тархалт Дүгнэлт"]
    ],
    on="Компани",
)

def select_best_fit(row):
    norm_p = row["Normal KS p-val"]
    log_p = row["Log-KS p-val"]
    
    if log_p > 0.05 and norm_p <= 0.05:
        return "Лог-хэвийн (Log-Normal)"
    elif norm_p > 0.05 and log_p <= 0.05:
        return "Хэвийн (Normal)"
    elif norm_p > 0.05 and log_p > 0.05:
        return "Лог-хэвийн (Log-Normal)" if log_p >= norm_p else "Хэвийн (Normal)"
    else:
        return "Бусад тархалт"

merged_df["Эцсийн Сонголт"] = merged_df.apply(select_best_fit, axis=1)


output_file = "Distribution_Test_Results.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    merged_df.to_excel(writer, sheet_name="Нэгтгэсэн Үр Дүн", index=False)
    norm_df.to_excel(writer, sheet_name="Хэвийн Тархалт", index=False)
    lognorm_df.to_excel(writer, sheet_name="Лог-Хэвийн Тархалт", index=False)

print(f"Үр дүн амжилттай '{output_file}' файл руу хадгалагдлаа.")
