import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle  # 用於保存模型

# ========== 配置區 ==========
DATA_PATH = "results/analyzed_data.csv"  # 從 analyze.py 生成的數據文件
MODEL_PATH = "results/stock_model.pkl"  # 保存訓練好的模型
RANDOM_SEED = 42  # 隨機種子
TEST_SIZE = 0.2  # 測試集比例

# ========== 1. 加載數據 ==========
def load_data(file_path):
    """
    加載分析後的數據
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"數據文件 {file_path} 不存在！")

# ========== 2. 數據處理 ==========
def preprocess_data(df):
    """
    處理數據，提取特徵和目標變量
    """
    # 填充缺失值
    df = df.ffill()  # 向前填充空值

    # 特徵選擇 (可以根據需求調整)
    features = ["MA5", "RSI", "Upper_Band", "Middle_Band", "Lower_Band", "Sentiment_Score"]
    target = "Price_At_Close"

    X = df[features].values
    y = df[target].values

    return X, y

# ========== 3. 訓練模型 ==========
def train_model(X, y):
    """
    訓練隨機森林回歸模型
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # 定義模型
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED)

    # 訓練模型
    model.fit(X_train, y_train)

    # 預測和評估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"模型性能: MSE = {mse:.2f}, MAE = {mae:.2f}")

    return model

# ========== 4. 保存模型 ==========
def save_model(model, file_path):
    """
    保存訓練好的模型到文件
    """
    with open(file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"模型已保存到 {file_path}")

# ========== 主函數 ==========
if __name__ == "__main__":
    # 加載數據
    df = load_data(DATA_PATH)
    
    # 處理數據
    X, y = preprocess_data(df)
    
    # 訓練模型
    model = train_model(X, y)
    
    # 保存模型
    save_model(model, MODEL_PATH)
