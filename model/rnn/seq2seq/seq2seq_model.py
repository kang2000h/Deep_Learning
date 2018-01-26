import random
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import *
from subtitle_converter import *

class Seq2SeqModel:
    """
    Sequence-to-sequence model with word embedding internally
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model shown in
    this page: https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py
    please look there for details, this code has made simply without nice skills 
    such as outputprojection&sampled softmax, multi buckets, 
    """
    #contructor for creaing the model
    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 hidden_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 isbe,
                 use_lstm = False,
                 num_samples = 512,
                 forward_only=False,
                 dtype = tf.float32):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(learning_rate, trainable = False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # as using output projection & sampled softmax
        output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocab size.
        if num_samples > 0 and num_samples < self.target_vocab_size:

            #print(self.target_vocab_size)
            #print(self.hidden_size)

            w_t = tf.get_variable("pro_w", [self.target_vocab_size, self.hidden_size], dtype=dtype) #for sampled_softmax_loss function
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)

            w = tf.transpose(w_t)  # to make outputprojection
            output_projection = (w, b)

            def sampled_loss(labels, logits): # original ver used tf.cast and took care of choosing a data type
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(
                    weights = w_t,
                    biases = b,
                    labels = labels, # on using ssl, it's labels is list of integer
                    inputs = logits,
                    num_sampled = num_samples,
                    num_classes = self.target_vocab_size )
            softmax_loss_function = sampled_loss



        # The seq2seq function : we only use embedding for input
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode): # just a tensor to be abstract
            with tf.variable_scope("embedding_rnn_seq2seq"):
                # Create the internal  cell for our RNN. # 셀을 생성하기 위해서 셀의 종류와 레이어의 개수를 통해서 사용할 함수 결정. 근데 텐서를 바로 반환
                if use_lstm:
                    def single_cell():
                        return tf.contrib.rnn.LSTMCell(hidden_size)
                else:
                    def single_cell():
                        return tf.contrib.rnn.GRUCell(hidden_size)
                if num_layers > 1:  # 여기도
                    cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
                else:
                    cell = single_cell()
            return embedding_attention_seq2seq( # embedding_rnn_seq2seq
                    encoder_inputs=encoder_inputs,
                    decoder_inputs=decoder_inputs,
                    cell=cell,
                    num_encoder_symbols = source_vocab_size,
                    num_decoder_symbols = target_vocab_size,
                    embedding_size = hidden_size,
                    output_projection = output_projection,
                    feed_previous = do_decode)

        # Feeds for inputs.
        self.encoder_inputs = [] # list of place holder to use ops in the session, len(encoder_input) * batch_size
        self.decoder_inputs = []
        self.target_weights = []

        for i in range(buckets[-1][0]): # maximum of length of the buckets,
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

        for i in range(buckets[-1][1] + 1): # 1 is for EOS tag
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i+1] for i in range(len(self.decoder_inputs) -1)] # START태그 말고 나머지 싹 가져옴.

        # Training outputs and losses
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets( # 객체 내부적으로 텐서를 가진다. 반환된 텐서가 bucket별로 나오네..-> 버킷별로 실행가능.
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function = softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.. it means..by doing this, we can get make-sense sentences
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [tf.matmul(output, output_projection[0])+output_projection[1]
                                       for output in self.outputs[b]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y : seq2seq_f(x, y, False),
                softmax_loss_function = softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only: # On training
            self.gradient_norms = [] # gradient_norms와 updates는 추이를 기록하기 위함으로 보임... 아직..
            self.updates = []
            #optimizer = tf.train.GradientDescentOptimizer(self.learning_rate) # 바꾸면서 최적화 하기.
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm) #gradient_norm에 대한 최대값은 초기화시킴.
                self.gradient_norms.append(norm)
                self.updates.append(optimizer.apply_gradients(
                    zip(clipped_gradients, params), global_step = self.global_step ))

        #self.saver = tf.train.Saver(tf.global_variables()) # 만든 그래프들에 대한 saver 객체를 내부적으로 초기화시킴.
        all_bar_key = tf.GraphKeys.GLOBAL_VARIABLES
        if isbe:
            vars_for_be = tf.get_collection(key=all_bar_key, scope="be_model")
        else :
            vars_for_be = tf.get_collection(key=all_bar_key, scope="gg_model")
        self.saver=tf.train.Saver(vars_for_be)

    # init에서 그래프 그리고, get_batch에서 데이터 받아서 돌림.
    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        """ Run a step of the model feeding the given inputs.
        
        :param session: tensorflow session to use. 
        :param encoder_inputs: list of numpy int vectors to feed as encoder inputs.
        :param decoder_inputs: list of numpy int vectors to feed as decoder inputs.
        :param target_weights: list of numpy float vectors to feed as target weights. [decoder_size * batch_size]
        :param bucket_id: which bucket of the model to use.
        :param forward_only: whether to do the backward wtep or only forward.
        :return: A triple consisting of gradient norm ( or None if we did not do backward), only forward니까 gradient는 필요없음.
                average perplexity, and the outputs.
        """
        # Check if the sizes match
        encoder_size, decoder_size = self.buckets[bucket_id] # 버킷사이즈와 비교.
        if len(encoder_inputs) != encoder_size: # 이게 그럼 step은 밖에서 이미 버킷 정하고 그에 맞는 입력만 넣었으니 비교 가능함.
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input Feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for idx in range(encoder_size):
            input_feed[self.encoder_inputs[idx].name] = encoder_inputs[idx] # fit to the shape of place_holder, encoder_size * batch_size
        for idx in range(decoder_size):
            input_feed[self.decoder_inputs[idx].name] = decoder_inputs[idx]
            input_feed[self.target_weights[idx].name] = target_weights[idx]

        # Since our targets are decoder inputs shifted by one, we need one more. 음?
        last_target = self.decoder_inputs[decoder_size].name

        #batch_size = len(encoder_inputs[0])
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed : depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for idx in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][idx])


        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None #??
        else:
            return None, outputs[0], outputs[1:] # we'll only use content part without losses[bucket_id]..

    # 데이터를 받아서 버킷에 따라 넣어 반환.
    def read_data(self, encoder_input, decoder_input, tdata_rate=0.8):
        enc_inputdata, dec_inputdata, enc_dict_key2val, dec_dict_val2key, _, _, _, _ = load_data_v2("./final/data_recent.csv",
                                                                                                    "./final/Encoder_Dict_key2val_file_forBE",
                                                                                                    "./final/Decoder_Dict_key2val_file_forBE",
                                                                                                    "./final/Decoder_Dict_val2key_file_forBE",
                                                                                                    "./final/Encoder_Dict_key2val_file_forGG",
                                                                                                    "./final/Decoder_Dict_key2val_file_forGG",
                                                                                                    "./final/Decoder_Dict_val2key_file_forGG")

        train_set = {bucket_id: [] for bucket_id in range(len(self.buckets))}
        test_set = {bucket_id: [] for bucket_id in range(len(self.buckets))}
        total_size = len(encoder_input)
        t_boundary = int(total_size*tdata_rate)
        train_data_enc = encoder_input[:t_boundary]
        train_data_dec = decoder_input[:t_boundary]
        test_data_enc = encoder_input[t_boundary:]
        test_data_dec = decoder_input[t_boundary:]

        # 버킷 사이즈에 들어가는 input, output을 튜플의 리스트로 버킷별로 넣는다.
        for enc, dec in zip(train_data_enc, train_data_dec):
            for bucket_id in range(len(self.buckets)):
                if self.buckets[bucket_id][0] >= len(enc) and self.buckets[bucket_id][1] >= len(dec):
                    train_set[bucket_id].append((enc, dec))
                    break

        for enc, dec in zip(test_data_enc, test_data_dec):
            for bucket_id in range(len(self.buckets)):
                if self.buckets[bucket_id][0] >= len(enc) and self.buckets[bucket_id][1] >= len(dec):
                    test_set[bucket_id].append((enc, dec))
                    break


        return train_set, test_set


    # get batch로 전처리 하기 전, 데이터를 준비할 때부터 batch_size는 고려됨. 그러므로 당연히 객체 만들기 전에 batch size를 초기화함..그럼 이 함수는 임의 버킷의 inputs length*batch size를 반환하는 것이 됨.
    # 데이터를 입력받되 step() 메소드를 사용하기 위한 batch-major로 데이터를 처리해서 반환해주는 메소드. 이건 bucket도 고려되야함.
    def get_batch(self, data, bucket_id):
        #{bucket_id : [(token_ids, [])]} : bucket_id로 참조하고 입출력 셋 튜플의 리스트.
        #난 그냥 입출력 셋을 튜플로 줄건데, 이 부분을 살피자. -> 버킷별로 나눠서 주는 정도는 read_data 메소드를 만들어서 위 형식으로 맞춰야 겠다.(버킷이 필요하니까. 그걸로 초기화한 객체에서 작업)
        #그리고 패딩, 태깅 하지말고 주자. load_data_v2를 만들어야할 듯 하다.
        data_size = len(data[bucket_id])

        encoder_size, decoder_size = self.buckets[bucket_id] # 해당 버킷의 크기
        encoder_inputs, decoder_inputs = [], []



        #for data_idx in range(data_size): # non minibatch
        for _ in range(self.batch_size):
            #encoder_input, decoder_input = data[bucket_id][data_idx]
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            PAD_ID = 0
            START_ID= 1

            encoder_pad_size = [PAD_ID]*(encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad_size)))

            # Decoder inputs get an extra <START> symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([START_ID] + decoder_input + [PAD_ID]*decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        #객체가 받은 batch_size쓰지말고 bucket_id에 든 인스턴스 개수를 넣어주면? 그래도 될거같은데..
        # Batch encoder inputs are just re-indexed encoder_inputs
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                #np.array([encoder_inputs[batch_idx][length_idx] for batch_idx in range(self.batch_size)], dtype=np.int32))
                np.array([encoder_inputs[data_idx][length_idx] for data_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs and we create weights
        for length_idx in range(decoder_size):

            batch_decoder_inputs.append(
                    #np.array([decoder_input[batch_idx][length_idx] for batch_idx in range(self.batch_size)], dtype=np.int32))
                    np.array([decoder_inputs[data_idx][length_idx] for data_idx in range(self.batch_size)], dtype=np.int32))


            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            #batch_weight = np.ones(data_size, dtype=np.float32)
            # for data_idx in range(data_size):
            for batch_idx in range(self.batch_size) :

                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size -1 : # if this iter is not the last time, gather the next one respectively to aware the last content of the batch
                    #target = decoder_inputs[batch_idx][length_idx+1]
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size -1 or target == "<PAD>": # 지금이 입력의 끝이거나 다음이 <PAD>라는 말은 지금이 <EOS>라는 거거든! 그러니까 0.
                    #batch_weight[batch_idx] = 0.0
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights # batch_weights는 decoder_size * batch_size

