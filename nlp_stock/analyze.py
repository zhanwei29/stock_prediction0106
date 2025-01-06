import pandas as pd
import talib

# 加載數據
data_path = "results/merged_news_data.csv"
df = pd.read_csv(data_path)

# 確保日期欄位正確格式化
df["Date"] = pd.to_datetime(df["Date"])

# 填充缺失值（向前和向後填充）
df["Price_At_Close"] = df["Price_At_Close"].ffill().bfill()

# 計算技術指標
# MA5
df["MA5"] = df.groupby("Ticker")["Price_At_Close"].transform(lambda x: x.rolling(window=5).mean())

# RSI
df["RSI"] = df.groupby("Ticker")["Price_At_Close"].transform(lambda x: talib.RSI(x, timeperiod=14))
df["RSI"] = df["RSI"].ffill().bfill()

# 布林帶
def calculate_bollinger_bands(prices):
    if len(prices) < 20:
        return pd.DataFrame({"Upper_Band": [None] * len(prices),
                             "Middle_Band": [None] * len(prices),
                             "Lower_Band": [None] * len(prices)})
    upper, middle, lower = talib.BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    return pd.DataFrame({"Upper_Band": upper, "Middle_Band": middle, "Lower_Band": lower})

# 計算布林帶，並排除分組列
bollinger_bands = df.groupby("Ticker", group_keys=False)["Price_At_Close"].apply(
    lambda prices: calculate_bollinger_bands(prices).reset_index(drop=True)
).reset_index(drop=True)

# 合併布林帶指標
df = pd.concat([df.reset_index(drop=True), bollinger_bands], axis=1)

# 移除無效值以計算相關性
clean_df = df.dropna(subset=["Sentiment_Score", "Price_At_Close", "MA5", "RSI"])

# 計算相關性
correlation_price = clean_df["Sentiment_Score"].corr(clean_df["Price_At_Close"])
correlation_ma5 = clean_df["Sentiment_Score"].corr(clean_df["MA5"])
correlation_rsi = clean_df["Sentiment_Score"].corr(clean_df["RSI"])

# 打印相關性結果
print(f"情感分數與收盤價的相關係數: {correlation_price:.2f}")
print(f"情感分數與 MA5 的相關係數: {correlation_ma5:.2f}")
print(f"情感分數與 RSI 的相關係數: {correlation_rsi:.2f}")

# 保存數據到文件
output_path = "results/analyzed_data.csv"
df.to_csv(output_path, index=False)
print(f"已將處理後的數據保存到 {output_path}")
