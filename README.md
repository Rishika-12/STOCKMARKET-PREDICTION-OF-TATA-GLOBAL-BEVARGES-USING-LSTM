# STOCKMARKET-PREDICTION-OF-TATA-GLOBAL-BEVARGES-USING-LSTM
#Prediction of TATA GLOBAL BEVARAGES for next 30 days.
#IMPORTING LIBRARIES
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
#READING THE DATA
df=pd.read_csv("/content/tgb capstone project dataset.csv")
df.head()
#DATA PREPROCESSING
df.isnull().sum()
df=df.dropna()
df.isnull().sum()
#VISUALISING DATA
plt.figure(figsize=(10,10))
plt.plot(df['Close'])
plt.title('Historical Stock Value')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()
#CREATING RETURN AND LOG RETURN
df['returns']=df.Close.pct_change()
df['log returns']=np.log(1+df['returns'])
df.dropna(inplace=True)
x=df[['Close','log returns']]
sc = MinMaxScaler(feature_range = (0, 1)).fit(x)
x_scaled=sc.transform(x)
y=[x[0] for x in x_scaled]
#SPLITTING INTO TEST AND TRAIN
split=int(len(x_scaled)*0.8)
x_train=x_scaled[:split]
x_test=x_scaled[split:len(x_scaled)]
y_train=y[:split]
y_test=y[split:len(y)]
assert len(x_train)==len(y_train)
assert len(x_test)==len(y_test)
#CREATING X AND Y TEST & TRAIN
n=100
xtrain=[]
ytrain=[]
xtest=[]
ytest=[]
for i in range(n,len(x_train)):
  xtrain.append(x_train[i-n:i,:x_train.shape[1]])
  ytrain.append(y_train[i])
for i in range(n,len(x_test)):
  xtest.append(x_test[i-n:i,:x_test.shape[1]])
  ytest.append(y_test[i])
value=np.array(ytrain[0])
value=np.c_[value,np.zeros(value.shape)]
sc.inverse_transform(value)
xtrain,ytrain=(np.array(xtrain),np.array(ytrain))
xtrain=np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2]))
print(xtrain.shape)
xtest,ytest=(np.array(xtest),np.array(ytest))
xtest=np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2]))
print(xtest.shape)
#CREATING LSTM MODEL
model=Sequential()
model.add(LSTM(4,input_shape=(xtrain.shape[1],xtrain.shape[2])))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(xtrain,ytrain,epochs=15,validation_data=(xtest,ytest),batch_size=16,verbose=1)
regressor.summary()
#PREDICTION
trainPredict=model.predict(xtrain)
testPredict=model.predict(xtest)
trainPredict=model.predict(xtrain)
testPredict=model.predict(xtest)
trainPredict=np.c_[trainPredict,np.zeros(trainPredict.shape)]
testPredict=np.c_[testPredict,np.zeros(testPredict.shape)]
trainPredict=np.c_[trainPredict,np.zeros(trainPredict.shape)]
testPredict=np.c_[testPredict,np.zeros(testPredict.shape)]
#INVERSE PREDICTIONS
trainPredict=sc.inverse_transform(trainPredict)
trainPredict=[x[0] for x in trainPredict]
testPredict=sc.inverse_transform(testPredict)
testPredict=[x[0] for x in testPredict]
#COMPARING TEST AND TRAIN PREDICTION
plt.figure(figsize=(16,10))
plt.plot(trainPredict, color = 'green', label = 'Train data PredictedTATA GLOBAL BEVARAGES Stock Price')
plt.plot(testPredict, color = 'red', label = 'Test data Predicted TATA GLOBAL BEVARAGES Stock Price')
plt.title(' TATA GLOBAL BEVARAGES Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('TATA GLOBAL BEVARAGES Stock Price')
plt.legend()
plt.show()
#PREDICTION FOR NEXT 5 DAYS 
print(trainPredict[:5])
print(testPredict[:5])
#PLOT  OF TEST VS TRAIN DATA PREDICTIONS FOR NEXT 5 DAYS
plt.figure(figsize=(16,10))
plt.plot(trainPredict[:5], color = 'green', label = 'Train data PredictedTATA GLOBAL BEVARAGES Stock Price')
plt.plot(testPredict[:5], color = 'red', label = 'Test data Predicted TATA GLOBAL BEVARAGES Stock Price')
plt.title(' TATA GLOBAL BEVARAGES Price Prediction for next 5 days')
plt.xlabel('Trading Day')
plt.ylabel('TATA GLOBAL BEVARAGES Stock Price')
plt.legend()
plt.show()
#PREDICTION FOR NEXT 10 DAYS
print(trainPredict[:10])
print(testPredict[:10])
#PLOT  OF TEST VS TRAIN DATA PREDICTIONS FOR NEXT 10 DAYS
plt.figure(figsize=(16,10))
plt.plot(trainPredict[:10], color = 'green', label = 'Train data PredictedTATA GLOBAL BEVARAGES Stock Price')
plt.plot(testPredict[:10], color = 'red', label = 'Test data Predicted TATA GLOBAL BEVARAGES Stock Price')
plt.title(' TATA GLOBAL BEVARAGES Price Prediction for next 10 days')
plt.xlabel('Trading Day')
plt.ylabel('TATA GLOBAL BEVARAGES Stock Price')
plt.legend()
plt.show()
#PREDICTION FOR NEXT 30 DAYS
print(trainPredict[:30])
print(testPredict[:30])
#PLOT OF TEST VS TRAIN DATA PREDICTIONS FOR NEXT 30 DAYS
plt.figure(figsize=(16,10))
plt.plot(trainPredict[:30], color = 'green', label = 'Train data PredictedTATA GLOBAL BEVARAGES Stock Price')
plt.plot(testPredict[:30], color = 'red', label = 'Test data Predicted TATA GLOBAL BEVARAGES Stock Price')
plt.title(' TATA GLOBAL BEVARAGES Price Prediction for next 30 days')
plt.xlabel('Trading Day')
plt.ylabel('TATA GLOBAL BEVARAGES Stock Price')
plt.legend()
plt.show()
