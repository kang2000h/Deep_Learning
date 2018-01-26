import codecs
from konlpy.tag import Twitter
from gensim.models import word2vec
import numpy as np
import tensorflow as tf

#Natural Language Process
#reading a type of strings, return listed morpheme lists for every row
def read_data(filename):
    with codecs.open(filename, encoding='utf-8-sig', mode='r') as f: #can't aware BOM on utf-16 if it is big or little endian
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

corpus_data = read_data('corpus2.txt')
print(corpus_data)

#return : tagged list of string 
def tokenize(doc): #spliting into morpheme
    tagger = Twitter()
    return ['/'.join(t) for t in tagger.pos(doc, norm=True, stem=True)]

train_docs = [row[0] for row in corpus_data]
print(train_docs)
sentences = [tokenize(d) for d in train_docs]
print(sentences)


model = word2vec.Word2Vec(sentences, min_count=1)
model.init_sims(replace=True)

#Reading data
train_data = read_data('train_data.txt')

#generating word vector using trained vocab
train_data = [row[0] for row in train_data]
train_data = [tokenize(d) for d in train_data]
print("Train Data")
print(train_data)

train_size = 20
wv_line=[]
sequence_length = []
word_vector=[] #total word vector

for i in range(train_size):
  for j in range(len(train_data[i])):
    wv_line.append(model[train_data[i][j]])
  sequence_length.append(len(train_data[i]))
  word_vector.append(wv_line)

word_vector = np.array(word_vector)
sequence_length = np.array(sequence_length)
print(sequence_length)

train_target = np.array(read_data('train_target.txt'))
print(train_target.shape)
train_target = np.reshape(train_target, train_size)
print("Train Target")
print(train_target.shape)

test_data =0
test_target =0


#-------Classification Model-----------------------------------------------------------------------------------------------------
learning_rate = 0.5
epochs = 100
total_data_size = 20

data_dim = 100
output_dim = 2
hidden_dim = 3
nb_classes = 2

x = word_vector
y = train_target

#split to train and testing
train_size = int(len(y)*0.7)
test_size = len(y) - train_size
trainX, testX = np.array(x[0:train_size],dtype=np.float32), np.array(x[train_size:len(x)])
trainY, testY = np.array(y[0:train_size],dtype=np.float32), np.array(y[train_size:len(y)])

#input placeholders
X = tf.placeholder(tf.float32, [None, None, data_dim],name = "input_X")
Y = tf.placeholder(tf.int32, [None],name = "input_Y")

#one_hot style for classification
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=sequence_length)

#FC layer
cf_w = tf.get_variable("cf_w", shape=[3, 10], initializer=tf.contrib.layers.xavier_initializer())
cf_b = tf.get_variable("cf_b", shape=[10], initializer=tf.contrib.layers.xavier_initializer())
fc_i = tf.add(tf.matmul(outputs[:,-1],cf_w),cf_b)

logits = tf.contrib.layers.fully_connected(fc_i, output_dim, activation_fn = None) 

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y_one_hot, logits = logits))

Y_pred = tf.nn.softmax(logits)
print("Y_pred")

# Minimize error using cross entropy
#cross_entropy = Y_one_hot*tf.log(Y_pred)
#loss = tf.reduce_mean(-tf.reduce_sum(cross_entropy,reduction_indices=1)) #2th param, direction to add 

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # training
    for i in range(epochs):
        avg_cost = 0.
        _= sess.run(train, feed_dict={X: x, Y: y})
        avg_cost += sess.run(tf.reduce_sum(loss), feed_dict={X: x, Y: y})/total_data_size
        if i % 10 == 0:
            print(i, avg_cost)
    print("training over")


    # Testing
    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), y) #2th param, direction to calculate
    print("Y_pred")
    print(sess.run(correct_prediction,  feed_dict={X:x, Y:y}))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy")
    print(accuracy.eval({X:x, Y:y}))

    #writer = tf.summary.FileWriter("./my_graph", sess.graph) 
    #writer.close()

















