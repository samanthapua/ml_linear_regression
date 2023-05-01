import pandas as pd
import pandas_ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
df = pd.read_csv('./GOOGL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Adj Close':'adj_close'})
df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
df.ta.ema(close='adj_close',length=10,append=True)
df = df.iloc[10:]
df = df[['adj_close','EMA_10']]
X_train, X_test, y_train, y_test = train_test_split(df[['adj_close']], df[['EMA_10']], test_size=.2)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Model coefficient: ', model.coef_)
print('Mean Absolute Error: ', mean_absolute_error(y_test,y_pred))
print('Coefficient of determination: ', r2_score(y_test,y_pred))# Calculate residuals

test_df = X_test.join(y_test)
ypred_df = pd.DataFrame(y_pred, columns=['predicted_value'])
test_df  = test_df.reset_index()
test_df = test_df.join(ypred_df)
df = df.reset_index()
df = pd.merge(df,test_df[['Date','predicted_value']],on='Date',how='left')
df = df[df['Date']>='2022-01-01']
years = mdates.YearLocator()
months = mdates.MonthLocator()
years_format = mdates.DateFormatter('%Y')
fig, ax = plt.subplots()
plt.title('Predicted Google Stock Prices (EMA10)',fontsize=12)
ax.plot(df.Date, df.adj_close,color='green',linewidth=2,linestyle='--',label='Actual Price')
ax.plot(df.Date, df.predicted_value,color='orange',linewidth=2,marker='.',label='Predicted Price')
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(years_format)
ax.xaxis.set_minor_locator(months)
ax.grid(color='grey')
ax.axis('equal')
ax.legend(loc='upper left')
ax.set_xlabel('Date')
plt.ylim(top=200) #ymax is your value
plt.ylim(bottom=0) #ymin is your value
plt.show()
plt.show()
