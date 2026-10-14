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
end_date = "2026-01-01"

# Жилийн хаалтын ханш хадгалах DataFrame
yearly_close = pd.DataFrame()

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
        # Жил бүрийн сүүлийн сарын хаалтын ханш
        yearly = data["Close"].resample("YE").last()
        yearly_close[company] = yearly

# Огноог багана болгох
yearly_close.index.name = "Date"
yearly_close = yearly_close.reset_index()

# Хоосон утгатай мөрүүдийг устгах
yearly_clean = yearly_close.dropna().copy()

# Ажиглалтын тоо
print("Анхны ажиглалт:", len(yearly_close))
print("Үлдсэн ажиглалт:", len(yearly_clean))
print("Хасагдсан мөр:", len(yearly_close) - len(yearly_clean))

# Цэвэрлэсэн өгөгдлийг Excel-д хадгалах
yearly_clean.to_excel(
    "yearly.xlsx",
    sheet_name="close",
    index=False
)
