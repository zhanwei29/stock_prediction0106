import pandas as pd
import pickle

# 加載模型
model_path = "results/stock_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)
print("模型加載成功！")

# 加載數據
data_path = "results/analyzed_data.csv"
df = pd.read_csv(data_path)
print(f"成功加載數據，行數: {len(df)}")

# 填補缺失值
df = df.ffill()  # 向前填補
df = df.bfill()  # 向後填補

# 按 Ticker 分組，計算平均特徵
grouped_data = df.groupby("Ticker").agg({
    "Sentiment_Score": "mean",
    "MA5": "mean",
    "RSI": "mean",
    "Upper_Band": "mean",
    "Middle_Band": "mean",
    "Lower_Band": "mean"
}).reset_index()

# 構建特徵向量
feature_columns = ["Sentiment_Score", "MA5", "RSI", "Upper_Band", "Middle_Band", "Lower_Band"]
X = grouped_data[feature_columns]

# 預測股價
predicted_prices = model.predict(X)
grouped_data["Predicted_Price"] = predicted_prices

# 保存結果
output_path = "results/stock_predictions.csv"
grouped_data.to_csv(output_path, index=False)
print(f"預測結果已保存到 {output_path}")
