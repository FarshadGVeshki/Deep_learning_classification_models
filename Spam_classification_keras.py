#%% Spam detection with keras
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#%%
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
import matplotlib.pyplot as plt

#%%
Data = pd.read_csv('Spam-Classification.csv')
Class_raw = Data["CLASS"]
TextMessage = Data["SMS"]


#%%
# tokenizer-lemmatizer
def customtokenize(str): 
    tokens=nltk.word_tokenize(str)
    nostop = list(filter(lambda token: token not in stopwords.words('english'), tokens))
    lemmatized=[lemmatizer.lemmatize(word) for word in nostop ]
    return lemmatized
# feature extraction
vectorizer = TfidfVectorizer(tokenizer=customtokenize) 
tfidf=vectorizer.fit_transform(TextMessage)
tfidf_array = tfidf.toarray()

label_encoder = preprocessing.LabelEncoder()
Class = label_encoder.fit_transform(Class_raw) # integer coding
Class = tf.keras.utils.to_categorical(Class,2) # one-hot coding
# datset split:
X_train,X_test,Y_train,Y_test = train_test_split(tfidf_array, Class, test_size=0.10)

#%% building the model
NB_CLASSES=2
N_HIDDEN=32
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_HIDDEN,input_shape=(X_train.shape[1],),name='Hidden-Layer-1',activation='relu'))
model.add(keras.layers.Dense(N_HIDDEN,name='Hidden-Layer-2',activation='relu'))
model.add(keras.layers.Dense(NB_CLASSES,name='Output-Layer',activation='softmax'))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
 
#%% training the model
BATCH_SIZE=256
EPOCHS=10
VALIDATION_SPLIT=0.2
history=model.fit(X_train,Y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,verbose=1,validation_split=VALIDATION_SPLIT)
pd.DataFrame(history.history)["accuracy"].plot(figsize=(8, 5))
plt.title("Accuracy improvements with Epoch")
plt.show()

print("\nEvaluation using the test Dataset :\n------------------------------------")
model.evaluate(X_test,Y_test)

#%% 
test_sms_1 = "We are pleased to inform you"
predict_tfidf=vectorizer.transform([test_sms_1]).toarray()
prediction=np.argmax(model.predict(predict_tfidf), axis=1 )
test_label_1 = label_encoder.inverse_transform(prediction)
print('text message:', test_sms_1, '; Label: ', test_label_1[0])

test_sms_2 = "OK let's see what happens"
predict_tfidf=vectorizer.transform([test_sms_2]).toarray()
prediction=np.argmax(model.predict(predict_tfidf), axis=1 )
test_label_2 = label_encoder.inverse_transform(prediction)
print('text message:', test_sms_2, '; Label: ', test_label_2[0])
