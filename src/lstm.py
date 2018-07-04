import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from src.dl_scripts import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json


# function to break data into batches
def batches(batch_size, data, labels):
    if len(data) != len(labels):
        raise Exception('Not Equal')
    id = np.arange(0, len(data))
    np.random.shuffle(id)
    data_batches = []
    labels_batches = []
    for i in range(0, len(data), batch_size):
        data_batch = data[i:i+batch_size]
        data_batches.append(data_batch)
        labels_batch = labels[i:i+batch_size]
        labels_batches.append(labels_batch)
    if len(data) % batch_size != 0:
        batch = data[i:len(data)]
        data_batches.append(batch)   
        labels_batch = labels[i:len(data)]
        labels_batches.append(labels_batch)
    return data_batches, labels_batches


data1 = pd.read_csv("data/spam.csv", encoding='latin-1')
data1 = data1.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data2 = pd.ExcelFile("data/indian.xls")
data2 = data2.parse('Sheet1')
data = pd.concat([data1, data2])


# Preparing data
max_features = 4000
msg_length = 160
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data['msg'].values)
X = tokenizer.texts_to_sequences(data['msg'])
X = pad_sequences(X, maxlen=msg_length, padding='pre')


# Let's save this out so we can use it later
dictionary = tokenizer.word_index
with open('saved/dictionary.json', 'w') as dictionary_file:
        json.dump(dictionary, dictionary_file)


Y = data['label'].values


# creating model
model = models.Model(max_features, msg_length)


# dividing for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
Y_train = pd.get_dummies(Y_train)


# metrics for evalution
prediction = tf.nn.softmax(model.out)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_train, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))


init = tf.global_variables_initializer()


# dividing into batches
batch_size = 500
data_batches, labels_batches = batches(batch_size, X_train, Y_train)


# feeding into model
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('results', sess.graph)
    epochs = 5
    print("started")
    tre = 1
    if len(data) % batch_size == 0:
        tre = 0

    for i in range(epochs):        
        for j in range(len(data_batches)):

            inp = model.train_inputs
            out = model.train_outputs
            sess.run([model.final_out], feed_dict={inp: data_batches[j],
                                                   out: labels_batches[j]})

        loss = sess.run(model.loss, feed_dict={inp: X_train, out: Y_train})
        acc = sess.run(accuracy, feed_dict={inp: X_train, out: Y_train})
        print("epoch: "+str(i)+" loss : "+str(loss)+" accuracy :"+str(acc))
    
    print("Saving.....") # saving the model
    inp_dict = {
        "train_inputs": inp,
        "train_outputs": out
    } 
    out_dict = {
        "final_out": model.out
    }
    tf.saved_model.simple_save(sess, "model_save", inp_dict, out_dict)

    Y_pred = sess.run(tf.argmax(prediction, 1), feed_dict={
                      model.train_inputs: X_test})
    
    writer.close()


Y_test = pd.get_dummies(Y_test, drop_first=True)
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


