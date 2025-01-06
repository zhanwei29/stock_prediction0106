import requests
import os
import shutil
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import logging
import yfinance as yf
from textblob import TextBlob  # 引入 TextBlob 用於情感分析

# 設置日誌記錄
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        logging.info(f"已清空資料夾: {directory}")
    os.makedirs(directory, exist_ok=True)


def fetch_stock_data(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker.upper()}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    os.makedirs("datasets", exist_ok=True)
    if response.status_code == 200:
        file_name = f"{ticker.lower()}_finviz.html"
        file_path = os.path.join("datasets", file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        logging.info(f"{ticker.upper()} 的 HTML 已成功保存到 {file_path}")
    else:
        logging.error(f"無法獲取 {ticker.upper()} 資料，HTTP 狀態碼: {response.status_code}")


def fetch_daily_closing_prices(ticker_list, start_date, end_date):
    all_prices = []
    for ticker in ticker_list:
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(start=start_date, end=end_date, interval="1d")
            if not hist.empty:
                hist.reset_index(inplace=True)
                hist["Date"] = hist["Date"].dt.date
                hist["Ticker"] = ticker.upper()
                all_prices.append(hist[["Ticker", "Date", "Close"]])
            else:
                logging.warning(f"無法獲取 {ticker} 的日級數據")
        except Exception as e:
            logging.error(f"獲取 {ticker} 的數據時出現錯誤: {e}")
    if all_prices:
        return pd.concat(all_prices, ignore_index=True)
    return pd.DataFrame(columns=["Ticker", "Date", "Close"])


def parse_html_to_dataframe(directory="datasets"):
    if not os.path.exists(directory):
        logging.error(f"{directory} 資料夾不存在，請先執行爬取步驟。")
        return None

    parsed_news = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        ticker = file_name.split("_")[0].upper()

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                soup = BeautifulSoup(file, "html.parser")
                news_table = soup.find(id="news-table")
                if not news_table:
                    logging.warning(f"未在 {file_name} 中找到 'news-table'，跳過該檔案。")
                    continue
                rows = news_table.findAll("tr")
                for row in rows:
                    if row.a and row.td:
                        headline = row.a.get_text(strip=True)
                        time_text = row.td.get_text(strip=True)
                        parsed_news.append([ticker, time_text, headline])
        except Exception as e:
            logging.error(f"解析 {file_name} 時出現錯誤: {e}")

    return pd.DataFrame(parsed_news, columns=["Ticker", "Time", "Headline"])


def process_time_column(df):
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    def convert_time(time_str):
        if "Today" in time_str:
            time_str = time_str.replace("Today", str(today))
        elif "Yesterday" in time_str:
            time_str = time_str.replace("Yesterday", str(yesterday))
        else:
            if len(time_str.split()) == 1:
                time_str = f"{today} {time_str}"

        for fmt in ("%Y-%m-%d %I:%M%p", "%b-%d-%y %I:%M%p"):
            try:
                return datetime.strptime(time_str, fmt).replace(tzinfo=None)
            except ValueError:
                continue
        return None

    df["Time"] = df["Time"].apply(convert_time)
    df.dropna(subset=["Time"], inplace=True)
    return df


def add_sentiment_scores(df):
    """
    新增一列情感分數，使用 TextBlob 分析新聞標題
    """
    def analyze_sentiment(headline):
        try:
            return TextBlob(headline).sentiment.polarity
        except Exception as e:
            logging.error(f"情感分析失敗: {e}")
            return 0

    df["Sentiment_Score"] = df["Headline"].apply(analyze_sentiment)
    return df


def match_news_with_closing_prices(news_df, closing_prices_df):
    news_df["Date"] = news_df["Time"].dt.date

    merged_df = pd.merge(news_df, closing_prices_df, on=["Ticker", "Date"], how="left")

    for i in range(len(merged_df)):
        if pd.isna(merged_df.loc[i, "Close"]):
            ticker = merged_df.loc[i, "Ticker"]
            news_date = merged_df.loc[i, "Date"]

            previous_prices = closing_prices_df[
                (closing_prices_df["Ticker"] == ticker) & (closing_prices_df["Date"] <= news_date)
            ]
            if not previous_prices.empty:
                merged_df.loc[i, "Close"] = previous_prices.iloc[-1]["Close"]

    merged_df.rename(columns={"Close": "Price_At_Close"}, inplace=True)
    return merged_df


def clean_and_save_data(df, output_path="results/news_with_closing_prices.csv"):
    """
    清理數據並儲存到 CSV。
    - 根據股票代號排序。
    - 根據時間排序。
    """
    df["Headline"] = df["Headline"].str.strip()
    df.drop_duplicates(subset=["Ticker", "Time", "Headline"], inplace=True)
    df["Price_At_Close"] = df["Price_At_Close"].round(2)

    df = df.sort_values(by=["Ticker", "Time"], ascending=[True, True])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"數據已保存到 {output_path}")


if __name__ == "__main__":
    clear_directory("datasets")
    clear_directory("results")

    ticker_input = input("請輸入多個股票代號（用逗號分隔，例如 TSLA,AAPL）：").strip()
    ticker_list = [ticker.strip() for ticker in ticker_input.split(",")]

    for ticker in ticker_list:
        fetch_stock_data(ticker)

    news_df = parse_html_to_dataframe()
    if news_df is not None:
        news_df = process_time_column(news_df)

        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        closing_prices_df = fetch_daily_closing_prices(ticker_list, start_date, end_date)

        if not closing_prices_df.empty:
            news_df = add_sentiment_scores(news_df)  # 增加情感分數
            matched_df = match_news_with_closing_prices(news_df, closing_prices_df)
            clean_and_save_data(matched_df, "results/news_with_closing_prices_with_sentiment.csv")
            logging.info("數據處理完成並保存！")
        else:
            logging.error("未能獲取有效的收盤價數據")