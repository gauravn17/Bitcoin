import pandas as pd
import pandas_ta as ta
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# Twitter API credentials (free tier)
consumer_key = 'ObThgAYyJlrSJU6O19biQ3aRg'
consumer_secret = 'pl3H4wXY1iMXPcgKl5OwWYGGrwZyAMuc5dr1bcUWCGRmRMyR3n'
access_token = 'y1820744999748726784-ZNOGiX8eqlpZoY2hD2XWRCknjlep9h'
access_token_secret = 'oIkSS51FZjz6i3k237BrIeLVq910DLlwzEZ1WqOclG50E'
#bearertoken = AAAAAAAAAAAAAAAAAAAAAE3ivAEAAAAAYmpNjEZkqBnhyZZHly4rAfARSJA%3D2AmuhuPb1lFxyB4CPVEzdklhfgk19J7oU9Yzwj1QuqGTwmE9Tc

# Set up Twitter API client
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def fetch_sentiment(date):
    query = 'Bitcoin'
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang='en', since=date, until=date).items(100)
    sentiment_scores = [sentiment_analyzer.polarity_scores(tweet.text)['compound'] for tweet in tweets]
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

# Fetch Bitcoin data
start_date = '2018-01-01'
end_date = '2023-12-02'
btc = yf.download('BTC-USD', start=start_date, end=end_date).reset_index()

# Calculate technical indicators
btc['SMA_20'] = ta.sma(btc['Close'], length=20)
btc['RSI'] = ta.rsi(btc['Close'], length=14)
btc['MACD'] = ta.macd(btc['Close'])['MACD_12_26_9']

# Fetch trading volume (already included in btc data)
btc['Volume'] = btc['Volume']

# Fetch social media sentiment
btc['Sentiment'] = btc['Date'].apply(lambda x: fetch_sentiment(x.strftime('%Y-%m-%d')))

# Fetch blockchain data
def fetch_transaction_volume(date):
    # Free API from CoinGecko
    endpoint = f'https://api.coingecko.com/api/v3/coins/bitcoin/history'
    params = {'date': date.strftime('%d-%m-%Y')}
    response = requests.get(endpoint, params=params)
    data = response.json()
    return data['market_data']['total_volume']['usd'] if 'market_data' in data else 0

btc['Transaction_Volume'] = btc['Date'].apply(lambda x: fetch_transaction_volume(x))

# Fetch macroeconomic indicators
sp500 = yf.download('^GSPC', start=start_date, end=end_date).reset_index()
sp500 = sp500.rename(columns={'Date': 'ds', 'Close': 'SP500_Close'})

# Merge S&P 500 data with btc data
btc = btc.merge(sp500[['ds', 'SP500_Close']], left_on='Date', right_on='ds', how='left').drop(columns=['ds'])

# Rename columns for Prophet
btc = btc.rename(columns={'Date': 'ds', 'Close': 'y'})

# Add technical indicators and other features as regressors
btc = btc[['ds', 'y', 'SMA_20', 'RSI', 'MACD', 'Volume', 'Sentiment', 'Transaction_Volume', 'SP500_Close']].dropna()

# Split data into training and test sets
train_size = int(len(btc) * 0.8)
train = btc[:train_size]
test = btc[train_size:]

# Initialize Prophet model
model = Prophet()
model.add_regressor('SMA_20')
model.add_regressor('RSI')
model.add_regressor('MACD')
model.add_regressor('Volume')
model.add_regressor('Sentiment')
model.add_regressor('Transaction_Volume')
model.add_regressor('SP500_Close')

# Fit model on training set
model.fit(train)

# Create dataframe with future dates for prediction
future = model.make_future_dataframe(periods=len(test))
future = future.merge(btc[['ds', 'SMA_20', 'RSI', 'MACD', 'Volume', 'Sentiment', 'Transaction_Volume', 'SP500_Close']], on='ds', how='left')

# Predict future values
forecast = model.predict(future)

# Extract forecast for test set period
test_forecast = forecast[forecast['ds'].isin(test['ds'])]

# Evaluate model
mae = mean_absolute_error(test['y'], test_forecast['yhat'])
mse = mean_squared_error(test['y'], test_forecast['yhat'])
rmse = mse ** 0.5

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot forecast
fig1 = model.plot(forecast)
plt.title('Bitcoin Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()

# Plot forecast components
fig2 = model.plot_components(forecast)
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(test['ds'], test['y'], label='Actual')
plt.plot(test_forecast['ds'], test_forecast['yhat'], label='Predicted')
plt.title('Actual vs Predicted Bitcoin Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()