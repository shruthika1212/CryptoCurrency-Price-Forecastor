# CryptoCurrency-Price-Forecastor
This is a cryptocurrency price forecasting application built using Python and Streamlit. It predicts the future price of various cryptocurrencies using historical data and Linear Regression. The application pulls data from Yahoo Finance and displays interactive visualizations of the price trends, predictions, and more.

Features
Select Cryptocurrency: Choose from popular cryptocurrencies such as Bitcoin (BTC), Ethereum (ETH), Dogecoin (DOGE), and more.
Prediction Days: Adjust the number of days for which the price will be predicted (from 1 to 30 days).
Price Prediction: View predicted future prices based on historical price data.
Data Visualization: Interactive graphs showing both raw historical data and predicted future prices.
Accuracy Check: Evaluate the model's prediction accuracy.
INR Converter: Convert the predicted price into Indian Rupees (INR).


Libraries and Technologies Used
Streamlit: For creating the interactive web application.
yfinance: To fetch historical cryptocurrency data from Yahoo Finance.
scikit-learn: For implementing Linear Regression to forecast future prices.
Plotly: For creating interactive and visually appealing charts.
NumPy and Pandas: For data manipulation and handling.
Matplotlib: For basic visualization (if needed).


How it Works
1.Data Fetching: The app fetches historical data for the selected cryptocurrency starting from January 1, 2019, until today's date using the yfinance library.
2.Data Preprocessing: It processes the data by shifting the 'Close' price to predict future prices.
3.Linear Regression Model: The app uses Linear Regression to predict the closing prices for the next few days based on historical data.
4.Visualization: Interactive graphs are displayed using Plotly, showing both the historical and predicted prices.

Project Structure
bash
Copy
Edit
.
├── app.py                  # Main Streamlit application file
├── requirements.txt         # List of dependencies for the project
├── README.md                # This file
