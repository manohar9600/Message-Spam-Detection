import pika
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import os
import numpy as np


# converting message to numerical value
max_features = 4000
msg_length = 160
tokenizer = Tokenizer(num_words=max_features)

with open('dictionary.json', 'r') as dictionary_file:
        tokenizer.word_index = json.load(dictionary_file)


# loading Tensorflow model
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = dir_path + '/model_save'

sess = tf.Session()
tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        dir_path
)


# functions
def detect_spam(msg):
    X = msg.split()
    X = tokenizer.texts_to_sequences(X)
    tem = []
    for num in X:
        if len(num) != 0:
            tem.append(num[0])
        else:
            tem.append(0)
    X = pad_sequences([tem], maxlen=msg_length, padding='pre')
    X = np.array(X).reshape(1, msg_length)

    prediction = sess.run(
        'Softmax:0',
        feed_dict={
            'Placeholder:0': X
        }
    )
    print(str(prediction[0][0]))

    return int(prediction[0][0] * 100)
    

# server code
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
channel.queue_declare(queue='rpc_queue')


def on_request(ch, method, props, body):

    msg = body.decode("utf-8") 
    print("spam " + msg)
    response = detect_spam(msg)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag = method.delivery_tag)


channel.basic_qos(prefetch_count=1)
channel.basic_consume(on_request, queue='rpc_queue')
print(" [x] Awaiting RPC requests")
channel.start_consuming()