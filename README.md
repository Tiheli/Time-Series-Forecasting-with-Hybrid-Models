# Time-Series-Forecasting-with-Hybrid-Models
Harika! Aşağıda, her adımı daha detaylı bir şekilde açıklayarak README dosyanız için uygun formatta hazırladım. Bu açıklamalar hem projenizin amacını hem de kullanılan adımların neden yapıldığını net bir şekilde ortaya koyacaktır.

---

## **Time Series Forecasting with Prophet, ARIMA, and LSTM**

### **Overview**
This project demonstrates how to apply time series forecasting techniques using three different models:
1. **Prophet** - A model designed for seasonality and trend-based forecasting.
2. **ARIMA** - A traditional statistical model for stationary time series.
3. **LSTM** - A neural network model that captures complex relationships in time series data.

The dataset consists of historical financial data (e.g., Bitcoin prices), and the objective is to compare the forecasting capabilities of these models.

---

### **Steps and Detailed Explanations**

#### **1. Installing Required Libraries**
```python
!pip install yfinance prophet statsmodels tensorflow
```
- **What we did:** Installed libraries like `yfinance` (for financial data), `prophet` (for trend-based forecasting), and `tensorflow` (for LSTM implementation).
- **Why:** These libraries provide essential tools for data collection, model implementation, and forecasting.

---

#### **2. Importing Libraries**
```python
from prophet import Prophet
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
```
- **What we did:** Imported libraries for data manipulation, visualization, and modeling.
- **Why:** Each library serves a specific purpose:
  - `yfinance`: To fetch historical financial data.
  - `Prophet`, `ARIMA`, `LSTM`: For forecasting.
  - `Matplotlib`: For visualization.

---

#### **3. Downloading Historical Data**
```python
df = yf.download('BTC-USD', "2016-01-01", "2024-12-18")
```
- **What we did:** Downloaded historical Bitcoin data from Yahoo Finance.
- **Why:** Time series forecasting requires sequential data, and Bitcoin prices serve as a relevant dataset.

---

#### **4. Data Preprocessing**
```python
df = df[['Close']]
df = df.reset_index()
```
- **What we did:** Selected the 'Close' column and reset the index for ease of manipulation.
- **Why:** Closing prices are often used in financial analysis as they reflect the final price for the day.

---

#### **5. Renaming Columns**
```python
df.columns = ['ds', 'y']
```
- **What we did:** Renamed columns to 'ds' (date) and 'y' (value) for Prophet compatibility.
- **Why:** Prophet expects these specific column names to process the data.

---

#### **6. Prophet Model Initialization**
```python
model = Prophet()
```
- **What we did:** Initialized the Prophet model.
- **Why:** Prophet is particularly effective for capturing seasonality and trends in time series data.

---

#### **7. Prophet Model Training**
```python
model.fit(df)
```
- **What we did:** Trained the Prophet model on historical data.
- **Why:** Training enables the model to learn patterns from the data.

---

#### **8. Prophet Forecasting**
```python
future = model.make_future_dataframe(periods=10)
myPredict = model.predict(future)
```
- **What we did:** Generated future predictions for the next 10 periods.
- **Why:** To forecast future values based on learned trends and seasonality.

---

#### **9. Prophet Visualization**
```python
model.plot(myPredict)
model.plot_components(myPredict)
```
- **What we did:** Visualized the overall forecast and its components (e.g., trend, seasonality).
- **Why:** Visualization provides insights into the model's understanding of the data.

---

#### **10. ARIMA Model Initialization**
```python
model_arima = ARIMA(df['y'], order=(5, 1, 0))
```
- **What we did:** Initialized an ARIMA model with specific parameters (p=5, d=1, q=0).
- **Why:** ARIMA is a statistical model for forecasting stationary time series.

---

#### **11. ARIMA Model Training**
```python
model_arima_fitted = model_arima.fit()
```
- **What we did:** Trained the ARIMA model.
- **Why:** Training allows the ARIMA model to learn dependencies and patterns.

---

#### **12. ARIMA Forecasting**
```python
forecast_arima = model_arima_fitted.forecast(steps=10)
```
- **What we did:** Generated future predictions using the trained ARIMA model.
- **Why:** To compare its performance with Prophet and LSTM.

---

#### **13. ARIMA Residual Analysis**
```python
residuals = model_arima_fitted.resid
plt.plot(residuals)
```
- **What we did:** Analyzed residuals to evaluate the model's accuracy.
- **Why:** Residual analysis helps diagnose model fit issues.

---

#### **14. LSTM Data Preparation**
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))
```
- **What we did:** Scaled data and prepared it for LSTM training.
- **Why:** Neural networks like LSTM require scaled inputs for efficient training.

---

#### **15. LSTM Model Initialization**
```python
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(units=1))
```
- **What we did:** Built an LSTM model with two layers and 50 units each.
- **Why:** LSTM is designed for capturing complex temporal dependencies in time series data.

---

#### **16. LSTM Model Training**
```python
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X, y, epochs=20, batch_size=32)
```
- **What we did:** Compiled and trained the LSTM model.
- **Why:** Training enables the LSTM model to learn temporal relationships.

---

#### **17. LSTM Forecasting**
```python
lstm_prediction = model_lstm.predict(X_test)
```
- **What we did:** Generated future predictions using the LSTM model.
- **Why:** To evaluate its performance alongside Prophet and ARIMA.

---

#### **18. Performance Comparison**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
```
- **What we did:** Calculated MAE, MSE, and RMSE for all models.
- **Why:** These metrics quantify model accuracy and enable fair comparisons.

---

#### **19. Visualization of Results**
```python
plt.bar(models, mae_values)
```
- **What we did:** Visualized the MAE values for all models.
- **Why:** Visualization simplifies understanding model performance differences.

---

#### **20. Logarithmic Scaling**
```python
plt.yscale('log')
```
- **What we did:** Applied a logarithmic scale to the y-axis.
- **Why:** This resolved issues with large value differences, ensuring all models were visible.

---

### **Usage**
Copy this README file to your GitHub repository for clear project documentation. If you need further help expanding it, feel free to ask! 😊
