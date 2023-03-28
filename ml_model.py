import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import csv
import requests
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
CSV_URL = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=IBM&interval=1min&slice=year1month1&apikey=UM3O1AZFWP2SZSTS'

with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    print(type(cr))
    my_list = list(cr)

    df = pd.DataFrame.from_records(my_list[1:])
    df.columns = my_list[0]
    print(df)
dataset_train=df

training_set= dataset_train.iloc[:,1:2].values

print(training_set.shape)


scaler =MinMaxScaler (feature_range = (0,1))
scaled_training_set = scaler.fit_transform(training_set)

X_train=[]
Y_train=[]

for i in range(60,training_set.shape[0]):
    X_train.append(scaled_training_set[i-60:i,0])
    Y_train.append(scaled_training_set[i,0])

X_train=np.array(X_train)
Y_train=np.array(Y_train)

X_train =np.reshape (X_train, (X_train.shape [0], X_train.shape[1], 1))


regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences= True, input_shape = (X_train.shape [1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM (units=50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences= True))
regressor.add(Dropout(0.2))

regressor.add(LSTM (units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))


url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=1min&apikey=UM3O1AZFWP2SZSTS%27'
r = requests.get(url)
data = r.json()

# Convert the dictionary into a list of dictionaries
time_series_list = []
for date, values in data["Time Series (1min)"].items():
    row_dict = {"date": date}
    for key, value in values.items():
        row_dict[key.split(". ")[1]] = value
    time_series_list.append(row_dict)

# Convert the list of dictionaries into a pandas DataFrame
time_series_df = pd.DataFrame(time_series_list)


# Convert the data types of the columns to float or int
time_series_df = time_series_df.astype({"open": float, "high": float, "low": float, "close": float, "volume": int})

# Print the final DataFrame
print(time_series_df)

dataset_test=time_series_df


actual_stock_price = dataset_test.iloc[:,1:2].values
print(actual_stock_price.shape)

dataset_total = pd.concat((dataset_train[ 'open'], dataset_test['open']), axis = 0)

inputs = dataset_total[len (dataset_total)- len(dataset_test)-60:].values

inputs = inputs.reshape(-1,1)

inputs =scaler.transform(inputs)

X_test = []

for i in range (60,100):
    X_test.append(inputs [i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape (X_test, (X_test.shape [0], X_test.shape [1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(actual_stock_price[60:], color = 'red', label = 'Actual IBM Stock Price')
plt.plot(predicted_stock_price,color = 'blue', label = 'Predicted IBM Stock Price')
plt.title('IBM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IBM Stock Price')
plt.legend()
