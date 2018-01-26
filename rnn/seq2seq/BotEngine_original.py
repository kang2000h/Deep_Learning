import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import *

from subtitle_converter import *

enc_inputdata, dec_inputdata, enc_dict_key2val, dec_dict_val2key ,_,_,_,_ = load_data("data_staryou.csv", "Encoder_Dict_key2val_file_forBE", "Decoder_Dict_key2val_file_forBE", "Decoder_Dict_val2key_file_forBE", "Encoder_Dict_key2val_file_forGG", "Decoder_Dict_key2val_file_forGG", "Decoder_Dict_val2key_file_forGG")

enc_inputdata = np.array(enc_inputdata)
dec_inputdata = np.array(dec_inputdata)
enc_sentence_length = 40 # 인코더 입력 최대 길이는 40, 이보다 작으면 <PAD>
dec_sentence_length = 40+1 # 디코더 입력 최대 길이는 42, 내용은 40이고 <START>
total_data_size = len(enc_inputdata)
batch_size = 50 # 테스트 데이터는 242개이다.

enc_vocab_size = len(enc_dict_key2val)
dec_vocab_size = len(dec_dict_val2key)

def idx2token(idx, reverse_vocab): #이제 반대로 인덱스를 받으면 값을 주는 메소드다. (아마 generative하게 인덱스를 추론하고 가능성 높은 녀석을 디코딩하는 작업에 필요하겠지)
    return reverse_vocab[str(idx)]

def idx2sent(indices, reverse_vocab=dec_dict_val2key): # 이제 숫자의 배열을 받으면 그녀석에 대한 워드 값들로 디코딩하고 " "(공백)으로 나눠서 붙일 것.
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])

n_epoch = 2000
#n_enc_layer = 3
#n_dec_layer = 3
hidden_size = 150 # output projection 적용해보자. 그렇다면.. (w, b)에서 w는 30*542 hidden_size*(len(dict)), b는 542 (len(dict)) 되야할 듯하다. 아니 그럼 딕셔너리가
num_cell=3

# enc_emb_size = hidden_size
#dec_emb_size = 50

###################
# 신경망 모델 구성#
###################
tf.reset_default_graph() #그래프 리셋, 반드시 현재 스레드에서 동작시켜야함.
########################################################################################################33
isTrain=False

enc_inputs = tf.placeholder(
    tf.int32,
    shape=[None, enc_sentence_length], # Batch_size * enc_sentence_length (시퀀스 길이) 즉, integer seq 받겠단 소리.
    name='input_sentences')

dec_inputs = tf.placeholder(
    tf.int32,
    shape=[None, dec_sentence_length],
    name='output_sentences')

# batch_major => time_major
enc_inputs_t = tf.transpose(enc_inputs, [1,0])
dec_inputs_t = tf.transpose(dec_inputs, [1,0])

#rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(hidden_size) for _ in range(num_cell)])

if isTrain==True :
    with tf.variable_scope("embedding_rnn_seq2seq"):
        # dec_outputs: [dec_sent_len+1 x batch_size x hidden_size] #디코더의 아웃풋은 EOS 붙여서 +1 해서 batch_size * dec_output_length가 될줄알았다. 근데 hidden_size 차원이 들어가네 ㅡㅡ 하긴 lstm의 output이 히든노드 수로 나오는게 맞네.
        dec_outputs, dec_last_state = embedding_rnn_seq2seq(  #허허 보통 tf로 임포트 안되네? 모듈을 따로 빼놔서 두개 받아야 했구나. 긜고, dec_output은 바로 시각별 출력으로 보면 되도록 되어있네., dec_output크기는 deccoder_inputs와 같다.
            encoder_inputs=tf.unstack(enc_inputs_t), # [batch_size],unpack : 주어진 디멘션을 빼내서 랭크를 R-1으로 만든다. value에서 num을 빼내는건데, num=None이면 axis의 축대로 빼낸 것들이 참조된다. default는 axis=0이니 0차원으로 자름..즉.. 배치별 시퀀스 순으로 받음.
            decoder_inputs=tf.unstack(dec_inputs_t), # [batch_size]
            cell=rnn_cell, #단층 구성했고.
            num_encoder_symbols=enc_vocab_size, # 단어 딕셔너리(중복X) + PAD태그
            num_decoder_symbols=dec_vocab_size, # 단어 딕셔너리(중복X) + GO, PAD태그. (근데,, 우선 끝까지 보자. UNK 태그 필요하지 않을지..!)
            embedding_size=hidden_size,
            feed_previous=False) # 훈련은 어쩌고 테스트 단계의 동작으로 만들어 둠? 우선 이렇게 잡아놔도 되네.. 의문은 든다. 제대로 하려면 Boolean형 텐서를 받아서 훈련과 테스트를 구분해야하지 않을까?
else :
    with tf.variable_scope("embedding_rnn_seq2seq"):
        dec_outputs, dec_last_state = embedding_rnn_seq2seq(
            encoder_inputs=tf.unstack(enc_inputs_t),
            decoder_inputs=tf.unstack(dec_inputs_t),
            cell=rnn_cell,
            num_encoder_symbols=enc_vocab_size,
            num_decoder_symbols=dec_vocab_size,
            embedding_size=hidden_size,
            feed_previous=True)


# predictions: [batch_size x dec_sentence_lengths+1]
predictions = tf.transpose(tf.argmax(tf.stack(dec_outputs), axis=-1), [1,0])

#labels & logits: [dec_sentence_length + 1 x batch_size x dec_vocab_size + 2]
labels = tf.one_hot(dec_inputs_t, dec_vocab_size)
logits = tf.stack(dec_outputs)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits))

# training_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
training_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss) # 최적화


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_history = []
    for epoch in range(n_epoch):

        all_inputs = []
        all_targets = []
        all_preds = [] # 에폭마다 예측값을 기록해둠.
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
            all_preds.append(batch_preds) # 예측한 정수 시퀀스를 저장해뒀다가 나중에 확인할 때 사용.


        # Logging every 400 epochs
        if epoch % 100 == 0: # 에폭을 2000은 했다니? 몇 가지 의문을 해결해보자.
            print('Epoch', epoch)
            for input_batch, target_batch, batch_preds in zip(all_inputs, all_targets, all_preds):
                for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                    print('\t', input_sent)
                    print('\t', pred)
                    print('\t => ', idx2sent(pred, reverse_vocab=dec_dict_val2key))
                    print('\tCorrent answer:', idx2sent(target_sent, reverse_vocab=dec_dict_val2key))
            print("\tepoch loss:",epoch_loss)

            #print('\tepoch loss: {:.2f}\n'.format(epoch_loss))
