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
    "SP500": "^GSPC"
}

# Хугацаа
start_date = "2012-01-01"
end_date = "2026-10-09"

# 7 хоногийн хаалтын ханш хадгалах DataFrame
weekly_close = pd.DataFrame()

# Компани бүрийн өгөгдлийг татах
for company, ticker in tickers.items():

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        multi_level_index=False,
        progress=False
    )

    if not data.empty:
        # Долоо хоногийн сүүлийн арилжааны өдрийн хаалтын ханш
        weekly = data["Close"].resample("W-FRI").last()
        weekly_close[company] = weekly

# Огноог багана болгох
weekly_close.index.name = "Date"
weekly_close = weekly_close.reset_index()

# Хоосон утгатай мөрүүдийг устгах
weekly_clean = weekly_close.dropna().copy()

# Ажиглалтын тоо
print("Анхны ажиглалт:", len(weekly_close))
print("Үлдсэн ажиглалт:", len(weekly_clean))
print("Хасагдсан мөр:", len(weekly_close) - len(weekly_clean))

# Цэвэрлэсэн өгөгдлийг Excel-д хадгалах
weekly_clean.to_excel(
    "weekly.xlsx",
    sheet_name="close",
    index=False
)
