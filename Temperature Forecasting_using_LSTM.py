import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DF=pd.read_csv("Weather_data.csv",index_col='datetime_utc',parse_dates=True)
temp=DF[' _tempm']
print(temp.shape)
print(temp.isnull().sum())
temp=temp.dropna()
print(temp.shape)

def df_to_X_y(df,size=5):
    df_as_np=df.to_numpy()
    X=[]
    y=[]
    for i in range(len(df_as_np)-size):
        row=[[a] for a in df_as_np[i:i+5]]
        X.append(row)
        label=df_as_np[i+5]
        y.append(label)
    return np.array(X),np.array(y)

X,y=df_to_X_y(temp,5)
print(X.shape,y.shape)

X_train,y_train=X[:70000],y[:70000]
X_val,y_val=X[70000:80000],y[70000:80000]
X_test,y_test=X[80000:],y[80000:]

print(X_train.shape,y_train.shape,X_val.shape,y_val.shape,X_test.shape,y_test.shape)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

model1 = Sequential()
model1.add(InputLayer((5, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model1.summary()
cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
model1.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=10, callbacks=[cp1])