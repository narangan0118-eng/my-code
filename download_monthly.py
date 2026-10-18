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
    "AMD": "AMD",
    "Tesla": "TSLA",
    "Intel": "INTC",
    "Broadcom": "AVGO",
    "SP500": "^GSPC"
}

# Хугацаа
start_date = "2012-05-01"
end_date = "2026-09-30"

# Сарын хаалтын ханш хадгалах DataFrame
monthly_close = pd.DataFrame()

# Компани бүрийн өгөгдлийг татах
for company, ticker in tickers.items():

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1mo",
        auto_adjust=True,
        multi_level_index=False,
        progress=False
    )

    if not data.empty:
        monthly_close[company] = data["Close"]

# Огноог багана болгох
monthly_close.index.name = "Date"
monthly_close = monthly_close.reset_index()

# Хоосон утгатай мөрүүдийг устгах
monthly_clean = monthly_close.dropna().copy()

# Ажиглалтын тоо
print("Анхны ажиглалт:", len(monthly_close))
print("Үлдсэн ажиглалт:", len(monthly_clean))
print("Хасагдсан мөр:", len(monthly_close) - len(monthly_clean))

# Цэвэрлэсэн өгөгдлийг Excel-д хадгалах
monthly_clean.to_excel(
    "monthly.xlsx",
    sheet_name="close",
    index=False
)
