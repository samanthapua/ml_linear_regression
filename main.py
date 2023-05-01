import pandas as pd
import pandas_ta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
df = pd.read_csv('./GOOGL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.rename(columns={'Adj Close':'adj_close'})

# df['EMA_10'] =
df.set_index(pd.DatetimeIndex(df['Date']), inplace=True)
df.ta.ema(close='adj_close',length=10,append=True)
df = df.iloc[10:]
df = df[['Date','adj_close','EMA_10']]
X_train, X_test, y_train, y_test = train_test_split(df[['adj_close']], df[['EMA_10']], test_size=.2)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Model coefficient: ', model.coef_)
print('Mean Absolute Error: ', mean_absolute_error(y_test,y_pred))
print('Coefficient of determination: ', r2_score(y_test,y_pred))# Calculate residuals
new_df = pd.DataFrame(y_pred, columns=['predicted_values'], index=df.index)
joined_df = df.join(new_df)
print(joined_df)
