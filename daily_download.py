import yfinance as yf
import pandas as pd

# 10 компанийн ticker
tickers = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Nvidia": "NVDA",
    "Meta": "META",
    "Samsung": "005930.KS",
    "Tesla": "TSLA",
    "Intel": "INTC",
    "Tencent": "0700.HK"
}

# Хугацаа
start_date = "2012-01-01"
end_date = "2026-10-09"

# Өдөр тутмын хаалтын ханш хадгалах DataFrame
daily_close = pd.DataFrame()

# Компани бүрийн өгөгдлийг татах
for company, ticker in tickers.items():

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        multi_level_index=False,
        progress=False
    )

    if not data.empty:
        daily_close[company] = data["Close"]

# Огноог багана болгох
daily_close.index.name = "Date"
daily_close = daily_close.reset_index()

# Хоосон утгатай мөрүүдийг устгах
daily_clean = daily_close.dropna().copy()

# Ажиглалтын тоо
print("Анхны ажиглалт:", len(daily_close))
print("Үлдсэн ажиглалт:", len(daily_clean))
print("Хасагдсан мөр:", len(daily_close) - len(daily_clean))

# Цэвэрлэсэн өгөгдлийг Excel-д хадгалах
daily_clean.to_excel(
    "daily.xlsx",
    sheet_name="close",
    index=False
)
