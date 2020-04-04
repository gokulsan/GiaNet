import pandas as pd
from fbprophet import Prophet

# Reading a File from GitHub

c = pd.read_csv("https://raw.githubusercontent.com/gokulsan/prophet/master/examples/example_wp_log_peyton_manning.csv")
print(c)

# Loading Facebook Prophet

m = Prophet()
m.fit(c)

# Setting up the Data Frame

future = m.make_future_dataframe(periods=365)
future.tail()

# Running Predictor 

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# Displaying Forecast

fig1 = m.plot(forecast)
