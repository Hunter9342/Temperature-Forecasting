import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DF=pd.read_csv("Weather_data.csv",index_col='datetime_utc',parse_dates=True)
temp=DF[' _tempm']
temp=temp.dropna()

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

X_train,y_train=X[:70000],y[:70000]
X_val,y_val=X[70000:80000],y[70000:80000]
X_test,y_test=X[80000:],y[80000:]


from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model

model1 = load_model('model1/')


train_predictions = model1.predict(X_train).flatten()
train_results = pd.DataFrame(data={'Train Predictions':train_predictions, 'Actuals':y_train})
print(train_results)


plt.plot(train_results['Train Predictions'][50:100])
plt.plot(train_results['Actuals'][50:100])
plt.show()

val_predictions = model1.predict(X_val).flatten()
val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val})
print(val_results)

plt.plot(val_results['Val Predictions'][:100])
plt.plot(val_results['Actuals'][:100])
plt.show()

test_predictions = model1.predict(X_test).flatten()
test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test})
print(test_results)

plt.plot(test_results['Test Predictions'][:100])
plt.plot(test_results['Actuals'][:100])
plt.show()


