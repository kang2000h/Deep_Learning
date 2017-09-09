from collections import Counter
import re

import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import *

from subtitle_converter import *

enc_inputdata, dec_inputdata, enc_dict_key2val, dec_dict_val2key ,_,_,_,_ = load_data("data_staryou.csv", "Encoder_Dict_key2val_file_forBE", "Decoder_Dict_key2val_file_forBE", "Decoder_Dict_val2key_file_forBE", "Encoder_Dict_key2val_file_forGG", "Decoder_Dict_key2val_file_forGG", "Decoder_Dict_val2key_file_forGG")

enc_inputdata = np.array(enc_inputdata)
dec_inputdata = np.array(dec_inputdata)
enc_sentence_length = 40 # 인코더 입력 최대 길이는 40, 이보다 작으면 <PAD>
dec_sentence_length = 40+1 # 디코더 입력 최대 길이는 42, 내용은 40이고 <START> 가 붙었거든.
total_data_size = len(enc_inputdata)
batch_size = 50 # 테스트 데이터는 242개이다.

enc_vocab_size = len(enc_dict_key2val)
dec_vocab_size = len(dec_dict_val2key)

'''
def token2idx(word, vocab): # dict가 있으면 토큰 받을때 정수로 바꿀 수 있다.
    return vocab[word]

#문장 하나를 받으면 인코딩 dict로 숫자를 만들어 낼수 있냐는 것. <PAD> 처리 이후 임베딩 처리 즉, 나는 bePadding, beEmbedded 사용하면 됨.
def sent2idx(sent, vocab=enc_vocab, max_sentence_length=enc_sentence_length, is_target=False):
    tokens = tokenizer(sent)
    current_length = len(tokens)
    pad_length = max_sentence_length - current_length
    if is_target:
        return [0] + [token2idx(token, vocab) for token in tokens] + [1] * pad_length #타겟(디코더 입력)이면 Go를 태그로 주고 토큰들 인덱스로 주고, 패딩 붙인다.
    else:
        return [token2idx(token, vocab) for token in tokens] + [0] * pad_length, current_length # 인코더 입력이면 시퀀스들을 정수 인덱스로 주고 패딩을 붙인다. 그리고 패딩을 제외한 길이를 같이 준다.





'''

def idx2token(idx, reverse_vocab): #이제 반대로 인덱스를 받으면 값을 주는 메소드다. (아마 generative하게 인덱스를 추론하고 가능성 높은 녀석을 디코딩하는 작업에 필요하겠지)
    return reverse_vocab[str(idx)]

def idx2sent(indices, reverse_vocab=dec_dict_val2key): # 이제 숫자의 배열을 받으면 그녀석에 대한 워드 값들로 디코딩하고 " "(공백)으로 나눠서 붙일 것.
    return " ".join([idx2token(idx, reverse_vocab) for idx in indices])

n_epoch = 2000
#n_enc_layer = 3
#n_dec_layer = 3
hidden_size = 100 # output projection 적용해보자. 그렇다면.. (w, b)에서 w는 30*542 (len(dict)), b는 542 (len(dict)) 되야할 듯하다. 아니 그럼 딕셔너리가 11만이면 output projection의 두번째가 11만개가 되야한단것? ㅡㅡ


enc_emb_size = 80 #임베딩 사이즈가 30이네..? 아마 딕셔너리에 비례해서 늘려야 할 듯하다. 차원수를 늘려야 의미를 분산시킬때 용이하니.
#dec_emb_size = 50 # 이 변수는 없어도 되는거 아님?

num_samples = 512

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
'''
sequence_lengths = tf.placeholder(
    tf.int32,
    shape=[None], # seq 길이만 받겠단 소리. 근데 왜 사용되는 곳이없지? 사실 기존 dynamic rnn같았으면 사용한다.
    name='sentences_length')
'''
dec_inputs = tf.placeholder(
    tf.int32,
    shape=[None, dec_sentence_length],
    name='output_sentences')


# batch_major => time_major
enc_inputs_t = tf.transpose(enc_inputs, [1,0]) #transpose는 a의 구조로부터 perm의 인덱스로 전치시킴. 즉, enc_sentence_length * batch_size 되었다.
dec_inputs_t = tf.transpose(dec_inputs, [1,0]) #(dec_sentence_length+1) * batch_size 되었다.

rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

w_t = tf.get_variable("proj_w", [dec_vocab_size, hidden_size], dtype=tf.float32)
w = tf.transpose(w_t)
b = tf.get_variable("proj_b", [dec_vocab_size], dtype=tf.float32)
output_projection = (w, b)

with tf.variable_scope("embedding_rnn_seq2seq"):
    # dec_outputs: [dec_sent_len+1 x batch_size x hidden_size] #디코더의 아웃풋은 EOS 붙여서 +1 해서 batch_size * dec_output_length가 될줄알았다. 근데 hidden_size 차원이 들어가네 ㅡㅡ 하긴 lstm의 output이 히든노드 수로 나오는게 맞네.
    dec_outputs, dec_last_state = embedding_rnn_seq2seq(  #허허 보통 tf로 임포트 안되네? 모듈을 따로 빼놔서 두개 받아야 했구나. 긜고, dec_output은 바로 시각별 출력으로 보면 되도록 되어있네., dec_output크기는 deccoder_inputs와 같다.
        encoder_inputs=tf.unstack(enc_inputs_t), # [batch_size],unpack : 주어진 디멘션을 빼내서 랭크를 R-1으로 만든다. value에서 num을 빼내는건데, num=None이면 axis의 축대로 빼낸 것들이 참조된다. default는 axis=0이니 0차원으로 자름..즉.. 배치별 시퀀스 순으로 받음.(왜 전치한거야?ㅡㅡ, 1축으로 자르기 싫었나?)
        decoder_inputs=tf.unstack(dec_inputs_t), # [batch_size]
        cell=rnn_cell, #단층 구성했고.
        num_encoder_symbols=enc_vocab_size, # 단어 딕셔너리(중복X) + PAD태그
        num_decoder_symbols=dec_vocab_size, # 단어 딕셔너리(중복X) + GO, PAD태그. (근데,, 우선 끝까지 보자. UNK 태그 필요하지 않을지..!)
        embedding_size=enc_emb_size, #내부적으로 분산표현을 사용하는데? 이거 워드2벡 말하는거야? 왜 하필 30이지? 디코더 딕셔너리 크기와 같게 넣었네. 그건 우연인듯하다. 심지어 10으로 테스트해도 됨.
        output_projection=output_projection,
        feed_previous=True) # 에에? 훈련은 어쩌고 테스트 단계의 동작으로 만들어 둠? 우선 이렇게 잡아놔도 되네.. 의문은 든다. 제대로 하려면 Boolean형 텐서를 받아서 훈련과 테스트를 구분해야하지 않을까?

#print("1",dec_outputs) # if output_projection is None, then the shape of outputs is ?*542, if output_projections is not None, then it shapes (? * 50)
# predictions: [batch_size x dec_sentence_lengths+1]
predictions = tf.transpose(tf.argmax(tf.stack(dec_outputs), axis=-1), [1,0]) # values 텐서들의 리스트를 받고 축에 대하여 쌓는다. 디폴트는 0차원 방향, 즉, outputs는 batch_size*seq_length*hidden_size 일건데, 뭐 N*hidden_size로 쌓이겠지.


#labels & logits: [dec_sentence_length + 1 x batch_size x dec_vocab_size + 2]
#labels = tf.one_hot(dec_inputs_t, dec_vocab_size) #default axis==-1, 즉, (dec_sentence_length+1)*batch_size*depth, 여기서 depth는 원핫표현시 크기(hidden_size)이며, 그것은 dec_vocab_size+2, 즉, 디코더용 데이터 딕셔너리 + GO, EOS 태그인듯.
#logits = tf.stack(dec_outputs) #레이블(타겟)과 비교할 예측값은 각 시각의 dec_output이다. 흠.. 왜 근데 이놈이 30인거야? 응 디코더 딕셔너리의 크기가 결국에는 30이거든, 소프트맥스 없고.. 사실상..넉넉하게 잡았다 봐야하나? 아니.. 그럼 딕셔너리가 30000개일때 노드를 30000만개 잡을순없잖아.ㅡㅡ?

#output projection 을 사용하면서 연산량을 줄이기 위해서 sampled softmax loss를 사용하고 이것은 training단계에서만 사용한단다. 일반적으로 훈련용으로 sampled softmax 사용하고 평가나 추론단계에서는 full softmax를 계산한단다.

#############################################################################
# 아니 그러니까, sampled softmax loss는 훈련때, 추론시에는 full softmax loss 사용란거잖아.
# 근데, 우리는 output projection은 사용해야하는 상황이야. hidden_size를 vocab에 맞출순없잖아. 즉 dec_output에서부터 logits을 가져와야하는데 그리고 경우에 따른 labels형태를 정해서 최종적으로 loss를 얻어내야한다.
# outputs를 받아서 output projection 시키고 결과를 훈련이냐 추론이냐에 따라 loss를 얻어내자.
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
        loss = tf.nn.sampled_softmax_loss( # 프로젝트가 들어가는거 맞네 그면,, num_classes가 vocab에 매핑시킬거니까, 이 웨이트로 output projection 할거고, dim은 hidden_size구만.
            weights=w_t, # weights: A Tensor of shape [num_classes, dim] 란다. 그래서 전치 시킴. 근데 왜 웨이트가 필요한거야? 소프트맥스는 정규화 레이어라 볼수있겠으나, 지금은 에러 펑션까지 합친 구조다. 그러니까 label과 input(logits)을 받느낟. 끝까지보자.
            biases=b,  # A Tensor of shape [num_classes]
            labels=labels,  # A Tensor of type int64 and shape [batch_size, num_true], not one-hot, num_true는 원핫으로부터의 인덱스인데, 그것은 딕셔너리 개수 만큼 가능하겠지.
            inputs=dec_outputs,  # A Tensor of shape [batch_size, dim]
            num_sampled=num_samples,  # An int. The number of classes to randomly sample per batch.
            num_classes=dec_vocab_size)  # An int. The number of possible classes.

        return tf.cast(loss ,dtype=tf.float32)
    else :
        labels = tf.one_hot(dec_inputs_t, dec_vocab_size)
        dec_outputs = tf.matmul(dec_outputs, w) + b
        logits = tf.stack(dec_outputs)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(  # batch_size 의 벡터만큼 로스가 반환될 것. 각 배치들의 로스들이겠지.
            labels=labels, # softmax cross entropy with logits의 lebels는 원핫이지만 sampled softmax loss는 integer라는 차이가 있다.
            logits=logits))

    return loss

#############################################################################
'''
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( # batch_size 의 벡터만큼 로스가 반환될 것. 각 배치들의 로스들이겠지.
    labels=labels, logits=logits)) # softmax cross entropy with logits의 lebels는 원핫이지만 sampled softmax loss는 integer라는 차이가 있다.
'''

if isTrain==True :
    loss = outputproject_n_get_loss(dec_inputs, dec_outputs, hidden_size, dec_vocab_size, isTrain=True)
else :
    loss = outputproject_n_get_loss(dec_inputs, dec_outputs, hidden_size, dec_vocab_size, isTrain=False)

#
# training_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
training_op = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss) # 최적화


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #그래프 초기화
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
            #데이터 준비되면 그걸로 훈련. (근데 여기선 얼마없는 데이터, 훈련할 때마다 싹 가져오네)
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

        # 100 에폭에 1회씩 성능평가를 할건데, 그때까지 쌓은 pred가 디코딩하면 어떤 값인지, 에러는 몇인지 확인함으로써 직관적으로 훈련과정을 확인해보는 것.
        # Logging every 400 epochs
        if epoch % 100 == 0: # 에폭을 2000은 했다니? 몇 가지 의문을 해결해보자.
            print('Epoch', epoch)
            for input_batch, target_batch, batch_preds in zip(all_inputs, all_targets, all_preds):
                for input_sent, target_sent, pred in zip(input_batch, target_batch, batch_preds):
                    print('\t', input_sent)
                    print('\t => ', idx2sent(pred, reverse_vocab=dec_dict_val2key))
                    print('\tCorrent answer:', idx2sent(target_sent, reverse_vocab=dec_dict_val2key))
            #print("\tepoch loss:",epoch_loss)
            #print("\tepoch loss:", len(epoch_loss))
            #print("\tepoch loss:", np.average(epoch_loss))

            print('\tepoch loss: {:.2f}\n'.format(epoch_loss))
