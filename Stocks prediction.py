import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler


training_data = pd.read_csv(r"D:\Stock prediction\Google_Stock_Price_Test.csv")
training_set = training_data.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# 'first_train' Input with 60 previous days' stock prices
first_train = []
# 'second_train' Output with next day's stock price
second_train = []
for i in range(60, 1258):
    first_train.append(training_set_scaled[i-60:i, 0])
    second_train.append(training_set_scaled[i, 0])
first_train, second_train = np.array(first_train), np.array(second_train)

# Reshaping (add more dimensions)
first_train = np.reshape(first_train, (first_train.shape[0], first_train.shape[1], 1))



# Initialising the RNN
# Regression is when you predict a continuous value
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# 'units' is the number of LSTM Memory Cells (Neurons) for higher dimensionality
# 'return_sequences = True' because we will add more stacked LSTM Layers
# 'input_shape' of first_train
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (first_train.shape[1], 1)))
# 20% of Neurons will be ignored (10 out of 50 Neurons) to prevent Overfitting
regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
# This is the last LSTM Layer. 'return_sequences = false' by default so we leave it out.
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# 'units = 1' because Output layer has one dimension
regressor.add(Dense(units = 1))

# Compiling the RNN
# Keras documentation recommends 'RMSprop' as a good optimizer for RNNs
# Trial and error suggests that 'adam' optimizer is a good choice
# loss = 'mean_squared_error' which is good for Regression vs. 'Binary Cross Entropy' previously used for Classification
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
# 'first_train' Independent variables
# 'second_train' Output Truths that we compare first_train to.
regressor.fit(first_train, second_train, epochs = 100, batch_size = 32)



# Getting the real stock price of 2017
testing_database = pd.read_csv("D:\Stock prediction\Google_Stock_Price_Test.csv")
actual_prices = testing_database.iloc[:, 1:2].values

# Combine 'training_data' and 'testing_database'
dataset_total = pd.concat((training_data['Open'], testing_database['Open']), axis = 0)
# Extract Stock Prices for Test time period, plus 60 days previous
inputs = dataset_total[len(dataset_total) - len(testing_database) - 60:].values
# 'reshape' function to get it into a NumPy format
inputs = inputs.reshape(-1,1)
# Inputs need to be scaled to match the model trained on Scaled Feature
inputs = sc.transform(inputs)
first_test = []

for i in range(60, 80):
    first_test.append(inputs[i-60:i, 0])

first_test = np.array(first_test)
# We need a 3D input so add another dimension
first_test = np.reshape(first_test, (first_test.shape[0], first_test.shape[1], 1))
# Predict the Stock Price
pred_prices = regressor.predict(first_test)
pred_prices = sc.inverse_transform(pred_prices)

#Plotting the results
plt.plot(actual_prices, color = 'red', label = 'Actual')
plt.plot(pred_prices, color = 'blue', label = 'Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()