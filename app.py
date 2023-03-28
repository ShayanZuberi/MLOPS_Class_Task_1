from flask import Flask
from keras.models import load_model
import requests
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as plt 
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict')
def preddict_stock():
    model = load_model('stock_predictor.h5' ,compile=False)
    scaler =MinMaxScaler (feature_range = (0,1))
    
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=1min&apikey=UM3O1AZFWP2SZSTS%27'
    r = requests.get(url)
    data = r.json()
    time_series_list = []
    for date, values in data["Time Series (1min)"].items():
        row_dict = {"date": date}
        for key, value in values.items():
            row_dict[key.split(". ")[1]] = value
        time_series_list.append(row_dict)
    time_series_df = pd.DataFrame(time_series_list)
    time_series_df = time_series_df.astype({"open": float, "high": float, "low": float, "close": float, "volume": int})
    dataset_test=time_series_df
    actual_stock_price = dataset_test.iloc[:,1:2].values
    scaled_training_set = scaler.fit_transform(actual_stock_price)
    inputs = dataset_test['open'].values

    inputs = inputs.reshape(-1,1)

    inputs =scaler.transform(inputs)

    X_test = []

    for i in range (60,100):
        X_test.append(inputs [i-60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape (X_test, (X_test.shape [0], X_test.shape [1], 1))

    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(predicted_stock_price, label = 'Predicted IBM Stock Price')
    axis.plot(actual_stock_price[60:], label = 'Actual IBM Stock Price')
    axis.set_xlabel('Time')
    axis.set_ylabel('IBM Stock Price')
    axis.legend()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

    #return predicted_stock_price.tolist()



