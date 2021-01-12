import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#df =
df.index.freq = None
df.head()
df.plot(figsize=(12,6),grid=True);
len(df)
train = df.iloc[:101]
test = df.iloc[101:]
print(train.shape, test.shape)
"""### Fit the model"""
from statsmodels.tsa.ar_model import AR,
ARResults
model = AR(train['Test'])
# Order 1 p=1 AR(1)
AR1fit =
model.fit(maxlag=1,method='cmle',trend='c',solver='l
bfgs')
# To know the order
AR1fit.k_ar
AR1fit.params
"""### Predict"""
start = len(train)
start
end=len(train) + len(test) - 1
end
pred1 = AR1fit.predict(start=start,end=end)
pred1 = pred1.rename('AR(1) Predictions')
test.plot(figsize=(12,6))
pred1.plot(legend=True);
# Order 2 p=2 AR(2)
AR2fit =
model.fit(maxlag=2,method='cmle',trend='c',solver='l
bfgs')
# Order
AR2fit.k_ar
# Parameters
AR2fit.params
pred2 = AR2fit.predict(start=start,end=end)
pred2 = pred2.rename('AR(2) Predictions')
pred2
test.plot(figsize=(12,6))
pred1.plot(legend=True)
pred2.plot(legend=True);
"""### Let Statsmodel choose the order for us"""
from statsmodels.tsa.ar_model import AR,
ARResults
model = AR(train['Test'])
ARfit = model.fit(ic='t-stat')
ARfit.k_ar # to know the right order
ARfit.params # to know all the parameters
pred11 = ARfit.predict(start=start,end=end)
pred11 = pred11.rename('AR(11) Predictions')
"""### Evaluate the model"""
from sklearn.metrics import mean_squared_error
labels = ['AR1','AR2','AR11']
preds = [pred1,pred2,pred11]
for i in range(3):
 error = mean_squared_error(test['Test'],preds[i])
 print('%s: Mean Squared Error =
%s'%(labels[i],error))
test.plot()
pred1.plot(legend=True)
pred2.plot(legend=True)
pred11.plot(legend=True)
plt.grid(True);
plt.savefig('AR_img.png');
# Forecast for Furture Values
model = AR(df['Test']) # Refit on the entire Dataset
ARfit = model.fit() # Refit on the entire Dataset
forecasted_values =
ARfit.predict(start=len(df),end=len(df)+36) #
Forecasting 3 year = 36 months
forecasted_values =
forecasted_values.rename('Forecast')
forecasted_values
# Plotting
df['Test'].plot(title='Forecasting for 36 Months')
forecasted_values.plot(legend=True)
plt.grid(True);
plt.savefig('AR_Forecast_img.png');