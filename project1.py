import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# putting historical data
ticker = 'WIT'  
data = yf.download(ticker, start='2014-01-01', end='2024-01-01')

# Making the data
data['Date'] = data.index
data = data[['Date', 'Open']]
data.reset_index(drop=True, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(pd.Timestamp.to_julian_date)

# Splitting data
X = data[['Date']]
y = data['Open']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# training the model to analyse
model = LinearRegression()
model.fit(X_train, y_train)

# making predictions
y_pred = model.predict(X_test)

# plotting the predictiobs
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Price Prediction')
plt.show()
