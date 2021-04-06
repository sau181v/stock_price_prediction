# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import pandas_datareader as web
# import datetime as dt
#
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, LSTM  # lstm-long short term memory layer
#
# # loading the data
# company = 'FB'
#
# start = dt.datetime(2012, 1, 1)  # at what time we wanna start our data from
# end = dt.datetime(2020, 1, 1)
#
# data = web.DataReader(company, 'yahoo', start, end)  # ticker symbol of companies...google it
#
# # prepare data for neural network
# # we are going to scale down all the value
# # so that they can fit between 0's and 1's
# scaler = MinMaxScaler(feature_range=(0, 1))  # this is from sklearn preprocessing modules
# scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))  # here i can use adjusted close....
# '''we are only gonna predict the closing price(after market is closed)'''
#
# # define prediction days
# prediction_days = 60
#
# # defining two empty lists
# x_train = []
# y_train = []
#
# for x in range(prediction_days, len(scaled_data)):  # we are gonna have 60 values
#     x_train.append(scaled_data[x-prediction_days:x, 0])
#     y_train.append(scaled_data[x, 0])
#
# x_train, y_train = np.array(x_train), np.array(y_train)  # we're converting those into numpy arrays
# x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#
# # building the models...here
# model = Sequential()
#
# # we're gonna add one dropout and 1 lstm layer in this fashion dense layer gonna
# # only one unit that's gonna be stock price prediction
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))  # prediction of the next price(closing value)
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=25, batch_size=32)
# # epochs meaning the model's gonna see the same data 24 times
# # batch_size meaning the models gonna see 32 units at once all the time
# # model.save()
#
# '''Test The model accuracy for existing data...'''
#
# # load the test data...(prepared)
# test_start = dt.datetime(2020, 1, 1)
# test_end = dt.datetime.now()
#
# test_data = web.DataReader(company, 'yahoo', test_start, test_end)
# # we need to get the prices, we need to scale the prices
# # we need to concatenate the full data set of the data we wanna predict on
# actual_prices = test_data['Close'].values  # real prices from stock market
#
# total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
# # total data combines training data to test data
#
# model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
# #  prediction_days: means up until the ends
# model_inputs = model_inputs.reshape(-1, 1)
# model_inputs = scaler.transform(model_inputs)
#
# # now we'll prediction on test data
#
# x_test = []
#
# for x in range(prediction_days, len(model_inputs)):
#     x_test.append(model_inputs[x-prediction_days:x, 0])
#
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#
# predicted_prices = model.predict(x_test)  # we are predicting the prices
# predicted_prices = scaler.inverse_transform(predicted_prices)  # we are inversing the predicted prices
#
# # .....plotting the Test predictions..............
# plt.plot(actual_prices, color="black", label=f"Actual {company} price")
# plt.plot(predicted_prices, color="green", label=f"Predicted {company} price")
# plt.title(f"{company} share price")
# plt.xlabel('Time')
# plt.ylabel(f'{company} share price')
# plt.legend()
# plt.show()
#
# #  predicting the next day (future days)
# real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
# real_data = np.array(real_data)
# real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
#
# # print(scaler.inverse_transform(real_data[-1]))
#
# prediction = model.predict(real_data)
# prediction = scaler.inverse_transform(prediction)
# print(f"Prediction : {prediction}")
