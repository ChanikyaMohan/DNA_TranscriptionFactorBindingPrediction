
# coding: utf-8

# In[1]:


from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential


# In[2]:


import pandas as pd
import numpy as np


# In[55]:


train_data = pd.read_csv("C:\\Users\\konya\\Desktop\\DNA_TFBP\\DNA_TranscriptionFactorBindingPrediction\\data\\train.csv");
test_data = pd.read_csv("C:\\Users\\konya\\Desktop\\DNA_TFBP\\DNA_TranscriptionFactorBindingPrediction\\data\\test.csv");
train_len = int(len(train_data))
print(train_len)


# In[56]:


#split train data into train and validation data
split_ratio = 0.8

train_size = int( train_len * (split_ratio))

#before taking partitions shuffle rows
#this is because all same class tuples are grouped together
train_data = train_data.sample(frac=1).reset_index(drop=False)

X_train = np.array(train_data['sequence'][0:])
Y_train = np.array(train_data['label'][0:])
X_test = np.array(train_data['sequence'][train_size:])
Y_test = np.array(train_data['label'][train_size:])


# In[57]:


#preprocessing of DNA sequences
from keras.preprocessing.text import Tokenizer
from keras import optimizers

tokenizer = Tokenizer(split='', char_level=True)
tokenizer.fit_on_texts(X_train)

print(tokenizer.word_counts)
print(tokenizer.document_count)
print(tokenizer.word_index)
print(tokenizer.word_docs)

encoded_X_train = tokenizer.texts_to_sequences(X_train) #mode for text_to_matrix = freq,count,binary,tfidf
print(encoded_X_train[0])


# In[58]:


embedding_matrix = np.zeros((5, 4))
embedding_matrix[1][0] = 1
embedding_matrix[2][1] = 1
embedding_matrix[3][2] = 1
embedding_matrix[4][3] = 1


# In[67]:


#LSTM model in keras
def model_init(input_length):
    LSTM_model = Sequential()
    print(input_length)
    LSTM_model.add(Embedding(5,4, weights=[embedding_matrix],
                            input_length=14,
                            trainable=False))
    
    LSTM_model.add(LSTM(activation="sigmoid", recurrent_activation='hard_sigmoid', return_sequences=True, units=256))
    LSTM_model.add(LSTM(activation="sigmoid", recurrent_activation='hard_sigmoid', units=256))
    LSTM_model.add(Dense(1, activation='sigmoid'))

    LSTM_model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['accuracy'])
    return LSTM_model


# In[60]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(X_test.shape)


# In[71]:


model = model_init(train_len)

trained_model = model.fit(np.array(encoded_X_train), np.array(Y_train), batch_size=20, epochs=15, validation_split = 0.2, verbose = 1)


# In[77]:


import matplotlib.pyplot as plt
print(trained_model.history.keys())

plt.plot(trained_model.history['acc'])
plt.plot(trained_model.history['val_acc'])
plt.title('model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(trained_model.history['loss'])
plt.plot(trained_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[78]:


encoded_X_test = tokenizer.texts_to_sequences(X_test)
score, accuracy = model.evaluate(np.array(encoded_X_test), np.array(Y_test), batch_size=1)
print('Score:', score)
print('Accuracy:', accuracy)


# In[32]:


from keras.models import model_from_json

model_JSON = model.to_json()
with open("C:\\Users\\konya\\Desktop\\DNA_TFBP\\DNA_TranscriptionFactorBindingPrediction\\models\\model_JSON.json","w") as json_file:
    json_file.write(model_JSON)
model.save_weights("C:\\Users\\konya\\Desktop\\DNA_TFBP\\DNA_TranscriptionFactorBindingPrediction\\models\\model_WEIGHTS.h5")
model.save("C:\\Users\\konya\\Desktop\\DNA_TFBP\\DNA_TranscriptionFactorBindingPrediction\\models\\model_model.h5")


# In[76]:


sub_X_test =  np.array(test_data['sequence'][0:])
encoded_sub_X_test = tokenizer.texts_to_sequences(sub_X_test)
#preds = model.predict(np.array(encoded_sub_X_test))
#print(preds[:10])
preds = model.predict_classes(np.array(encoded_sub_X_test))

sub_df = pd.DataFrame(data=preds,columns={"prediction"})
sub_df.to_csv(path_or_buf="C:\\Users\\konya\\Desktop\\DNA_TFBP\\DNA_TranscriptionFactorBindingPrediction\\results\\sub.csv", columns={"prediction"}, header=True, index=True, index_label="id")
print("submission file ready!")


# In[35]:


#load saved model and predict
# load json and create model
json_file = open("C:\\Users\\konya\\Desktop\\DNA_TFBP\\DNA_TranscriptionFactorBindingPrediction\\models\\model_JSON.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:\\Users\\konya\\Desktop\\DNA_TFBP\\DNA_TranscriptionFactorBindingPrediction\\models\\model_WEIGHTS.h5")
print("Loaded model from disk")

#predict using loaded model
preds_loaded = loaded_model.predict_classes(np.array(encoded_sub_X_test))
print(preds_loaded[:10])

