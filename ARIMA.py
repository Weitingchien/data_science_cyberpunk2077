import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 模擬數據集，正常情況下這裡將使用您的評論數據
# timestamp代表時間戳，count代表該時間的評論數量
data = {
    'timestamp': pd.date_range(start='2022-11-01', periods=300, freq='D'),
    'count': np.random.randint(10, 100, size=300)  # 假設的評論數量
}
df = pd.DataFrame(data)
df.set_index('timestamp', inplace=True)

# 選擇ARIMA模型參數
# p是自回歸階數，d是差分階數，q是移動平均階數
p, d, q = 2, 1, 2
model = ARIMA(df['count'], order=(p, d, q))

# 擬合模型
model_fit = model.fit()

# 進行預測
df['forecast'] = model_fit.predict(start=0, end=len(df)-1, typ='levels')

# 繪圖顯示原始數據和預測數據
plt.figure(figsize=(10,5))
plt.plot(df['count'], label='Original')
plt.plot(df['forecast'], label='Forecast')
plt.title('ARIMA Model Forecast')
plt.legend()
plt.show()