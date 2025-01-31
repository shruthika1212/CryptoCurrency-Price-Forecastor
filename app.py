import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
from datetime import date
import yfinance as yf 
from plotly import graph_objs as go

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Cryptocurrency Price Forecaster")
html_temp = """
    <marquee behavior="scroll" direction="left">ALL INVESTMENTS ARE SUBJECT TO PRICE FLUCTUATIONS AND OTHER MARKET RISKS..... </marquee>
    """
st.markdown(html_temp,unsafe_allow_html=True)

currency=("ETH-USD","BTC-USD","BNB-USD","MATIC-USD","TRX-USD","DOGE-USD","SOL-USD","ATOM-USD")
selected_currency=st.selectbox("select coin",currency)

n_days = st.slider("Days of prediction",1,30)

# period =n_years*365

#@st.cache(allow_output_mutation=True)

# @st.cache
def load_data(ticker):
    data=yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state=st.text("load data...")
data=load_data(selected_currency)
data_load_state.text("loading data....done")

# st.subheader('Raw data')
# st.write(data.head(7))
# st.write(data.tail(7))

def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='close'))
    fig.layout.update(title_text="Time Series Graph", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
# plot_raw_data()
 
def get_data(ticker):
    data=load_data(ticker)
    data[str(n_days)+'_Day_Price_Forecast'] = data[['Close']].shift(-n_days)
    X= np.array(data[['Close']])
    X= X[:data.shape[0]-n_days]
    y= np.array(data[str(n_days)+'_Day_Price_Forecast'])
    y= y[:-n_days]
    return X,y

X, y= get_data(selected_currency)

#linear regression
def result(X,y):
    X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2) 
    linReg = LinearRegression()
    linReg.fit(X_train,y_train)
    x_projection = np.array(data[['Close']])[-n_days:]
    linReg_prediction = linReg.predict(x_projection)
    lr_acc = linReg.score(X_test,y_test)#r^2 test
    return x_projection,linReg_prediction,lr_acc
n, m, p= result(X,y)

# st.write(n)
# st.subheader('Predicted prices ')
# st.write(m)

r_list=list(range(1,n_days))
def plot_result_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=r_list, y=m, name='predicted'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
# plot_result_data()
if st.button("View Past data"):

    #st.subheader('Raw data')
    st.write(data.head(7))
    st.write(data.tail(7))
    plot_raw_data()

if st.button("Predict future Prices"):
    st.subheader('Predicted prices ')
    st.write(m)
    plot_result_data()
if st.button("Accuracy check"):
    st.write(p*100)
if st.button('INR CONVERTER'):
    st.write(m*82.56)
    st.write(f'''
        <a target="_blank" href="https://www.coinbase.com/learn/crypto-basics">
            <button style = "background-color:#16767B; border-radius:7px;">
                LEARN MORE
            </button>
        </a>
        ''',
        unsafe_allow_html=True
    )

