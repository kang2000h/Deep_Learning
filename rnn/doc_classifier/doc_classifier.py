import codecs

import os
import numpy as np
import tensorflow as tf

from make_wordvectors import tokenize, input_data, make_wordvectors
from konlpy.tag import Twitter
from gensim.models import word2vec
import random

import RNN_Graphs

#훈련 데이터 읽기
dir_list=[['c1'], ['c2']] #class1, class2
c1_sents=[]
c2_sents=[]
NumOfFile = 1000
word2vec_dim = 50

#[c1_sents, c2_sents] = input_data(dir_list, 1000, "cp949" )
data = input_data(dir_list, NumOfFile, "cp949" )

#make_wordvectors('c1_c2_word2vec_model2',data, 1000)


#훈련된 vocab으로 word vector 생성
model = word2vec.Word2Vec.load('c1_c2_word2vec_model3')
model.init_sims()
#[str] = tokenize('정부')
#print(str)
#print(model.wv.most_similar(positive=[str]))

sequence_length = []

sentences = [c1_sents,c2_sents]
for label in range(len(data)):
    for EachFile in range(NumOfFile):
        EachSentence = data[label][EachFile]
        ListOfWords = tokenize(EachSentence)
        for word in ListOfWords:
            # if word=="./Punctuation":
            if 'Punctuation' in word:
                ListOfWords.remove(word)
            elif 'Josa' in word:
                ListOfWords.remove(word)
            elif 'KoreanParticle' in word:
                ListOfWords.remove(word)
            elif '하다/Verb' in word:
                ListOfWords.remove(word)
            elif '되다/Verb' in word:
                ListOfWords.remove(word)
            elif '들/Suffix' in word:
                ListOfWords.remove(word)
            elif '적/Suffix' in word:
                ListOfWords.remove(word)
        sequence_length.append(len(ListOfWords))
        sentences[label].append(ListOfWords)

print(len(sequence_length))
print("한글워드 전처리 완료")
sentences = np.array(sentences) #(2, 1000)
print(sentences.shape)
print(sentences[0][0])

max_seq_len = 1000
word_vector=[] #total word vector
#WordEmbedding & if the length of input sequence is over 1000, just use them
for lable in range(len(data)):
    for file_i in range(NumOfFile):
        wv_line = []
        sent_len=len(sentences[lable][file_i])
        if sent_len ==0:
            print(file_i,"th file is empty")
        if sent_len > max_seq_len:
            sent_len = max_seq_len

        for word_i in range(sent_len):
            wv_line.append(model[sentences[lable][file_i][word_i]])
        for word_i in range(sent_len, max_seq_len):
            wv_line.append(np.zeros_like(wv_line[0])) #For padding
        sequence_length[lable*NumOfFile+file_i]=sent_len
        word_vector.append(wv_line)

word_vector = np.array(word_vector)
print(word_vector.shape)
sequence_length = np.array(sequence_length)
print(sequence_length.shape)
print(np.max(sequence_length))

target_data=[]
for i in range(NumOfFile*2):
    if i < NumOfFile:
        target_data.append(0)
    else :
        target_data.append(1)


target_data = np.array(target_data)
print("훈련데이터 셋 완료")

#--------
# ---------------------------------------
# --------------------------------
learning_rate = 0.0003

epochs = 20
total_data_size = 2000
batch_size = 200

data_dim = 50
hidden_dim = 100 # larger than 200, Resc2rceExhaustedError 

FCL1_c2tput_dim= 200
FCL2_c2tput_dim = 40
last_c2tput_dim = 5
nb_classes = 2
train_size = int(NumOfFile*0.7) #the size of train data
num_cells=1

x = word_vector
y = target_data

# Define weights
weights = {
    #'W1': tf.Variable(tf.random_normal([hidden_dim, last_c2tput_dim]))
    'W1': tf.get_variable("w1", shape=[hidden_dim, last_c2tput_dim],initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b1': tf.Variable(tf.random_normal([last_c2tput_dim]))
}
#W2 = tf.Variable(np.random.rand(hidden_dim, last_c2tput_dim), dtype=tf.float32)
#b2 = tf.Variable(np.zeros((1, last_c2tput_dim)), dtype=tf.float32)

#split to train and testing
X_class0 = x[:0*NumOfFile+NumOfFile]
X_class1 = x[0*NumOfFile+NumOfFile:1*NumOfFile+NumOfFile]
Seq_class0 = sequence_length[:0*NumOfFile+NumOfFile]
Y_class0 = y[:0*NumOfFile+NumOfFile]
Y_class1 = y[0*NumOfFile+NumOfFile:1*NumOfFile+NumOfFile]
Seq_class1 = sequence_length[0*NumOfFile+NumOfFile:1*NumOfFile+NumOfFile]

trainX =  np.concatenate((X_class0[:train_size],X_class1[:train_size]), axis=0)
trainY = np.concatenate((Y_class0[:train_size], Y_class1[:train_size]), axis=0)
train_seq = np.concatenate((Seq_class0[:train_size], Seq_class1[:train_size]),axis=0)
testX = np.concatenate((X_class0[train_size:], X_class1[train_size:]),axis=0)
testY = np.concatenate((Y_class0[train_size:], Y_class1[train_size:]),axis=0)
test_seq = np.concatenate((Seq_class0[train_size:], Seq_class1[train_size:]),axis=0)


#input placeholders
X = tf.placeholder(tf.float32, [batch_size, max_seq_len, data_dim])
Y = tf.placeholder(tf.int32, [batch_size])
seqLen = tf.placeholder(tf.int32, [batch_size])
phase_train = tf.placeholder(tf.bool, name='phase_train')

#one_hot style for classification
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

#cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim)
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_dim) for _ in range(num_cells)])
c2tputs, current_state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seqLen) # c2tput : [batch_size, seq*, hidden_dim]


#어떻게 정확히 마지막 출력 시각 인덱스를 가져올까. # c2tput series를 tf변수로 잡으면 배치 사이즈에 종속한 코드방식를 피할수있지 않을까 배치 사이즈를 가져올 수 있지 않을까.
#batch_series = tf.unstack(c2tputs)
#c2tput_series = batch_series[0][sequence_length[0]]
#c2tput_series = tf.gather(batch_series, 0)


c2tput_series=[]
for batch_idx in range(batch_size):
   #index = tf.range(0, batch_size) * max_seq_len + (seqLen - 1)
   last_c2tput_idx = tf.gather(seqLen, batch_idx) -1
   c2tput_series.append(c2tputs[batch_idx][last_c2tput_idx])

c2tput_series = tf.reshape(c2tput_series, [-1, hidden_dim]) # [batch_size, hidden_dim] * [hidden_dim, last_c2tput_dim]
rnn_c2tput = tf.matmul(c2tput_series, weights['W1']) + biases['b1'] # rnn_c2tput : [batch_size, last_c2tput_dim]
#rnn_c2tput = tf.matmul(c2tputs[:,-1], W2) + b2

rnn_c2tput_mean, rnn_c2tput_val = tf.nn.moments(rnn_c2tput,[0])
scale1 = tf.Variable(tf.ones([last_c2tput_dim]))
beta1 = tf.Variable(tf.zeros([last_c2tput_dim]))
epsilon = 1e-3
normed_rnn_c2tput = tf.nn.batch_normalization(rnn_c2tput,rnn_c2tput_mean,rnn_c2tput_val,beta1,scale1,epsilon)


fcl_1 = tf.contrib.layers.fully_connected(normed_rnn_c2tput, FCL1_c2tput_dim) # fcl_1 : [batch_size, FCL_c2tput_dim]
fcl_1_mean, fcl_1_val = tf.nn.moments(fcl_1,[0])
scale2 = tf.Variable(tf.ones([FCL1_c2tput_dim]))
beta2 = tf.Variable(tf.zeros([FCL1_c2tput_dim]))
normed_fcl_1 = tf.nn.batch_normalization(fcl_1,fcl_1_mean,fcl_1_val,beta2,scale2,epsilon)


fcl_2 = tf.contrib.layers.fully_connected(normed_fcl_1, FCL2_c2tput_dim) # fcl_1 : [batch_size, FCL_c2tput_dim]
fcl_2_mean, fcl_2_val = tf.nn.moments(fcl_2,[0])
scale3 = tf.Variable(tf.ones([FCL2_c2tput_dim]))
beta3 = tf.Variable(tf.zeros([FCL2_c2tput_dim]))
normed_fcl_2 = tf.nn.batch_normalization(fcl_2,fcl_2_mean,fcl_2_val,beta3,scale3,epsilon)


logits = tf.contrib.layers.fully_connected(normed_fcl_2, nb_classes) #? Softmax & CE
Y_pred = tf.nn.softmax(logits)

#Cross entropy cost/loss
cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y_one_hot, logits=logits)
avg_cost = tf.reduce_mean(cost)

#optimizer
optimizer = tf.train.AdagradOptimizer(learning_rate)
train = optimizer.minimize(avg_cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(Y_pred,1), tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

indice=np.zeros(1400)
with tf.Session() as sess:
    sess.run(init)

    # chkpt_file = '../Classification_data/classifiation_lstm.ckpt'
    #chkpt_file = "/Classification_data/classifiation_lstm"
    # /tmp/model.ckpt
    save_path = 'Classification_data/'
    model_name = 'classifiation_lstm'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path_full = os.path.join(save_path, model_name)

    # training
    for i in range(epochs):
        acc=0.0
        loss=0.0
        for hop in range(7):
            batch_mask = np.random.choice(1400, batch_size)
            #batch_mask = range(hop*batch_size, (hop+1)*batch_size)
            #bathc_mask = np.array(batch_mask)
            trainx = trainX[batch_mask]
            trainy = trainY[batch_mask]
            train_seq_ = train_seq[batch_mask]
            _, acc, avg_cost_, Y_, y_one_hot = sess.run([train, accuracy, avg_cost, Y_pred, Y_one_hot], feed_dict={X: trainx, Y: trainy, seqLen: train_seq_})

            for hypo in range(len(Y_)): #200
                if Y_[hypo][0] == 0.5 :
                    indice[batch_mask[hypo]] +=1
            print(indice)

        print("epoch: ", i+1, ", Loss = ", avg_cost_, ", Accuracy : ", acc)
        print("Estimation = ", Y_[:10], y_one_hot[:10])

    print("Optimization Finished!")
    np.save("test_indice2", indice)

    valacc_list=[]
    # Testing
    for step in range(3):
        testx = testX[step*batch_size:(step+1)*batch_size]
        testy = testY[step*batch_size:(step+1)*batch_size]
        test_seq_ = test_seq[step*batch_size:(step+1)*batch_size]
        valacc_list.append(accuracy.eval({X: testx, Y: testy, seqLen: test_seq_}))
        print("Testing Accuracy:", valacc_list[step])

    print(np.sum(valacc_list)/len(valacc_list))

    save_path = saver.save(sess, save_path_full)
    print("Model saved in file: %s" % save_path)