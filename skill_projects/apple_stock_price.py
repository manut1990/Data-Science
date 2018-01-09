###This project is based on Siraj Raval's YouTube Tutorial: "Predicting Stock Prices - Learn Python for Data Science #4"
'''You can find the video here: https://www.youtube.com/watch?v=SSu00IRRraY
However, I had to do it differently as I had to use quandl as datasource instead of a Google Finance csv-file and also made changes to the approach.


The goal here is to create a machine learning model, that predicts stock prices on the bases of the stock prices of the last 30, 60 or 90 days
Therefore, I used SVM regression (linear, polynomial and RBF). RBF turns out to be best fitting model though overfitting might be an issue with this model.
'''
import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt7
from sklearn.svm import SVR


x= quandl.get_table('WIKI/PRICES', ticker='AAPL', qopts={'columns': ['close', 'date']})

#dates = []
#prices = []
ticker='AAPL'

def get_data(ticker):
    df = quandl.get_table('WIKI/PRICES', ticker=ticker, qopts={'columns': ['close', 'date']})
    dates_quandl, prices_quandl = df['date'], df['close']
    dates_quandl.to_pickle('dates_quandl.pickle')
    prices_quandl.to_pickle('prices_quandl.pickle')
    return None

def predict_prices(dates, prices, x):
    dates= [i for i in range(0, len(dates))]
    dates = np.array(dates)
    prices = np.array(prices)
    dates = dates.reshape((len(dates), 1))
    prices = prices.reshape((len(prices), 1))
     
    #load SVR models; low C for noisy data; gamma for rbf high as this means data is very complex
    svr_lin = SVR(kernel= 'linear', C= 1e3)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    
    #Fit SVR models
    svr_lin.fit(dates, prices)
    svr_rbf.fit(dates, prices)
    
    #plot models
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_rbf.predict(dates), color='orange', label='RBF model')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return svr_lin.predict(x)[0], svr_rbf.predict(x)[0], 

#Helper function to get the right amount of days
def get_data_for_x_days(dates, prices, days):
    return list(dates)[-days:], list(prices)[-days:]



##1 get data
##1.1 get data from quandl and pickle data; this only has to be done the first time running the code,
##    as data will be stored as pickle.
get_data(ticker)

##1.2 get data from pickle
dates_quandl = pd.read_pickle('dates_quandl.pickle')
prices_quandl = pd.read_pickle('prices_quandl.pickle')

##1.3 get dates and prices of the last 30, 60 and 90 days
dates_30, prices_30 = get_data_for_x_days(dates_quandl, prices_quandl, 30)
dates_60, prices_60 = get_data_for_x_days(dates_quandl, prices_quandl, 60)
dates_90, prices_90 = get_data_for_x_days(dates_quandl, prices_quandl, 90

##2 start model #print out model prediction for last day of sample and use model function to model, fit and plot different models
predicted_price_30 = predict_prices(dates_30, prices_30, 29)  
predicted_price_60 = predict_prices(dates_60, prices_60, 59)  
predicted_price_90 = predict_prices(dates_90, prices_90, 89)  
print(predicted_price_30, predicted_price_60, predicted_price_90)


