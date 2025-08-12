# 投資.py
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from matplotlib import rcParams
import tensorflow as tf
import os

# ---------- 中文字體 ----------
rcParams['font.sans-serif'] = ['Microsoft JhengHei']
rcParams['axes.unicode_minus'] = False

# ---------- 參數檢查 ----------
if len(sys.argv) < 3:
    print("用法: python 投資.py 股票代碼 預測天數")
    print("範例: python 投資.py 2330.TW 365")
    sys.exit(1)

symbol = sys.argv[1]
predict_days = int(sys.argv[2])
look_back = 60

# ---------- GPU / 混合精度 ----------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        mixed_precision.set_global_policy('mixed_float16')
        print("GPU 偵測到，已啟用混合精度。")
    except Exception as e:
        print("混合精度設定失敗。", e)
else:
    print("未偵測到 GPU，使用 CPU 訓練。")

# ---------- 下載資料 ----------
print(f"下載 {symbol} 過去 3 年 OHLCV 資料...")
df = yf.download(symbol, period='3y', interval='1d', auto_adjust=False)

# 修正 Close 欄位
if 'Close' not in df.columns:
    if 'Adj Close' in df.columns:
        print("未找到 Close 欄位，改用 Adj Close。")
        df = df.rename(columns={'Adj Close': 'Close'})
    else:
        raise ValueError("資料中沒有 Close 或 Adj Close 欄位！")

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# ---------- 技術指標 ----------
df['MA5'] = df['Close'].rolling(5).mean()
df['MA10'] = df['Close'].rolling(10).mean()
df['MA20'] = df['Close'].rolling(20).mean()
delta = df['Close'].diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
rs = roll_up / (roll_down + 1e-9)
df['RSI'] = 100 - (100 / (1 + rs))
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df = df.dropna()

# ---------- 縮放 ----------
feature_cols = list(df.columns)
close_col_idx = feature_cols.index('Close')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.values)

# ---------- 建立序列 ----------
X, Y = [], []
for i in range(look_back, len(scaled_data) - predict_days + 1):
    X.append(scaled_data[i - look_back:i, :])
    Y.append(scaled_data[i:i + predict_days, close_col_idx])
X = np.array(X)
Y = np.array(Y)

# ---------- 模型 ----------
n_features = X.shape[2]
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(look_back, n_features)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(predict_days, dtype='float32')
])
model.compile(optimizer=Adam(learning_rate=1e-3), loss=Huber(delta=1.0))
model.summary()

# ---------- 訓練 ----------
epochs = 80
batch_size = 32
print(f"開始訓練：epochs={epochs}, batch_size={batch_size}")
model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)

# ---------- 儲存模型 ----------
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"{symbol}_lstm_model.h5")
model.save(model_path)
print(f"模型已儲存：{model_path}")

# ---------- 預測 ----------
last_seq = scaled_data[-look_back:, :].reshape(1, look_back, n_features)
pred_scaled_close_seq = model.predict(last_seq, verbose=0).flatten()
pred_scaled_full = np.tile(scaled_data[-1, :], (predict_days, 1))
pred_scaled_full[:, close_col_idx] = pred_scaled_close_seq
inv_full = scaler.inverse_transform(pred_scaled_full)
inv_preds = inv_full[:, close_col_idx].flatten()

# ---------- 儲存預測結果 ----------
future_dates = pd.bdate_range(df.index[-1], periods=predict_days + 1)[1:]
pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": inv_preds
})
pred_dir = "predictions"
os.makedirs(pred_dir, exist_ok=True)
pred_path = os.path.join(pred_dir, f"{symbol}_pred_{predict_days}d.csv")
pred_df.to_csv(pred_path, index=False)
print(f"預測結果已儲存：{pred_path}")

# ---------- 畫圖 ----------
hist_prices = df['Close'].values.flatten()
plt.figure(figsize=(14,7))
plt.plot(df.index, hist_prices, label='歷史收盤價', color='tab:blue')
plt.plot(future_dates, inv_preds, label=f'預測 {predict_days} 日', color='tab:orange')
plt.scatter(future_dates, inv_preds, color='red', s=20)
plt.axvline(df.index[-1], color='green', linestyle='--', label='預測起點')
plt.title(f'{symbol} LSTM 多特徵 Seq2Seq 預測 ({predict_days} 日)')
plt.xlabel('日期')
plt.ylabel('價格')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
