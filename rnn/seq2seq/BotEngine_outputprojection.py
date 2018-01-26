import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import *

from subtitle_converter import *

enc_inputdata, dec_inputdata, enc_dict_key2val, dec_dict_val2key ,_,_,_,_ = load_data("data_staryou.csv", "Encoder_Dict_key2val_file_forBE", "Decoder_Dict_key2val_file_forBE", "Decoder_Dict_val2key_file_forBE", "Encoder_Dict_key2val_file_forGG", "Decoder_Dict_key2val_file_forGG", "Decoder_Dict_val2key_file_forGG")

enc_inputdata = np.array(enc_inputdata)
dec_inputdata = np.array(dec_inputdata)
enc_sentence_length = 40 # 인코더 입력 최대 길이는 40, 이보다 작으면 <PAD>
dec_sentence_length = 40+1 # 디코더 입력 최대 길이는 42, 내용은 40이고 <START> 
total_data_size = len(enc_inputdata)
batch_size = 50 # 테스트 데이터는 242개

enc_vocab_size = len(enc_dict_key2val)
dec_vocab_size = len(dec_dict_val2key)


def idx2token(idx, reverse_vocab): #이제 반대로 인덱스를 받으면 값을 주는 메소드다. (아마 generative하게 인덱스를 추론하고 가능성 높은 녀석을 디코딩하는 작업에 필요하겠지)
    return reverse_vocab[str(idx)]

def idx2sent(indices, reverse_vocab=dec_dict_val2key): # 이제 숫자의 배열을 받으면 그녀석에 대한 워드 값들로 디코딩하고 " "(공백)으로 나눠서 붙일 것.
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])

n_epoch = 2000
hidden_size = 100 #


enc_emb_size = 80 

num_samples = 512

###################
# 신경망 모델 구성#
###################
tf.reset_default_graph() 
########################################################################################################33
isTrain=False

enc_inputs = tf.placeholder(
    tf.int32,
    shape=[None, enc_sentence_length], 
    name='input_sentences')
dec_inputs = tf.placeholder(
    tf.int32,
    shape=[None, dec_sentence_length],
    name='output_sentences')


# batch_major => time_major
enc_inputs_t = tf.transpose(enc_inputs, [1,0]) 
dec_inputs_t = tf.transpose(dec_inputs, [1,0]) 

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

w_t = tf.get_variable("proj_w", [dec_vocab_size, hidden_size], dtype=tf.float32)
w = tf.transpose(w_t)
b = tf.get_variable("proj_b", [dec_vocab_size], dtype=tf.float32)
output_projection = (w, b)

with tf.variable_scope("embedding_rnn_seq2seq"):
    
    dec_outputs, dec_last_state = embedding_rnn_seq2seq( 
        encoder_inputs=tf.unstack(enc_inputs_t), 
        decoder_inputs=tf.unstack(dec_inputs_t), 
        cell=rnn_cell, 
        num_encoder_symbols=enc_vocab_size, 
        num_decoder_symbols=dec_vocab_size, 
        embedding_size=enc_emb_size, 
        output_projection=output_projection,
        feed_previous=True) 


predictions = tf.transpose(tf.argmax(tf.stack(dec_outputs), axis=-1), [1,0]) 






#############################################################################

def outputproject_n_get_loss(dec_inputs, dec_outputs, hidden_size, dec_vocab_size, isTrain=True):
    w_t = tf.get_variable("proj_w_forloss", [dec_vocab_size, hidden_size], dtype=tf.float32)
    w = tf.transpose(w_t)
    b = tf.get_variable("proj_b_forloss", [dec_vocab_size], dtype=tf.float32)
    output_projection = (w, b)
    #print(dec_outputs) #?*542
    dec_outputs = tf.reshape(dec_outputs, [-1, hidden_size])
    #print(dec_outputs) #?*50
    labels = tf.reshape(dec_inputs, [-1, 1])

    if isTrain :
        loss = tf.nn.sampled_softmax_loss( 
            weights=w_t, 
            biases=b,  
            labels=labels,  
            inputs=dec_outputs, 
            num_sampled=num_samples, 
            num_classes=dec_vocab_size) 

        return tf.cast(loss ,dtype=tf.float32)
    else :
        labels = tf.one_hot(dec_inputs_t, dec_vocab_size)
        dec_outputs = tf.matmul(dec_outputs, w) + b
        logits = tf.stack(dec_outputs)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(  
            labels=labels, 
            logits=logits))

    return loss

#############################################################################

if isTrain==True :
    loss = outputproject_n_get_loss(dec_inputs, dec_outputs, hidden_size, dec_vocab_size, isTrain=True)
else :
    loss = outputproject_n_get_loss(dec_inputs, dec_outputs, hidden_size, dec_vocab_size, isTrain=False)

#
# training_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
training_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss) 


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    loss_history = []
    for epoch in range(n_epoch):

        all_inputs = []
        all_targets = []
        all_preds = []
        epoch_loss = 0.0
        for _ in range(5):
            batch_mask = np.random.choice(total_data_size, batch_size)

            input_token_indices = enc_inputdata[batch_mask]
            target_token_indices = dec_inputdata[batch_mask]

            all_inputs.append(input_token_indices)
            all_targets.append(target_token_indices)
            
            # Evaluate three operations in the graph
            # => predictions, loss, training_op(optimzier)
            batch_preds, batch_loss, _ = sess.run(
                [predictions, loss, training_op],
                feed_dict={
                    enc_inputs: input_token_indices,
                    dec_inputs: target_token_indices
                })
            loss_history.append(batch_loss)
            epoch_loss += batch_loss
            all_preds.append(batch_preds) 

        
        # Logging every 400 epochs
        if epoch % 100 == 0: 
            print('Epoch', epoch)
            for input_batch, target_batch, batch_preds in zip(all_inputs, all_targets, all_preds):
                for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                    print('\t', input_sent)
                    print('\t => ', idx2sent(pred, reverse_vocab=dec_dict_val2key))
                    print('\tCorrent answer:', idx2sent(target_sent, reverse_vocab=dec_dict_val2key))
            #print("\tepoch loss:",epoch_loss)
            print('\tepoch loss: {:.2f}\n'.format(epoch_loss))
