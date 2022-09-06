#%% Iris classification with keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%%
Data = pd.read_csv("iris.csv")
label_encoder = preprocessing.LabelEncoder() # integer encoding
Data["Species"] = label_encoder.fit_transform(Data["Species"])
npData = Data.to_numpy()
X = npData[:,0:4]
Y = npData[:,4]
Xscaler = StandardScaler().fit(X) # data normalization (centering and scaling)
X = Xscaler.transform(X)
Y = tf.keras.utils.to_categorical(Y,3) # one-hot encoding
[X_train, X_test, Y_train, Y_test] = train_test_split( X, Y, test_size = 0.1)

#%% building the model
num_class = 3
num_nod_hid_lay_1 = 128
num_nod_hid_lay_2 = 128
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(num_nod_hid_lay_1, input_shape=(4,),name='hid_lay_1', activation='relu'))
model.add(keras.layers.Dense(num_nod_hid_lay_2,name='hid_lay_2', activation='relu'))
model.add(keras.layers.Dense(num_class,name='out_lay', activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
#%%  training the model
history = model.fit(X,Y,batch_size=16,epochs=10,verbose=1,validation_split=0.2)
pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
plt.title("Accuracy in each epoch")
plt.show()
model.evaluate(X_test,Y_test)

#%%  Prediction using the trained model
z = [[0.3, 1.2, 5.3, 4]] # a sample
print("z=", z)
z_scaled = Xscaler.transform(z) # scaled using the transform learned from training data
z_prediction = model.predict(z_scaled)
print("z label probablities:", z_prediction)
z_prediction = np.argmax(z_prediction)
z_label = label_encoder.inverse_transform([z_prediction])
print("z label:", z_label)