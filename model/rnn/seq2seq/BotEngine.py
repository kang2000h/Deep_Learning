import time
import math

from subtitle_converter import *
import utils
import tensorflow as tf

#seq2seq_model 안에 정의된 Seq2SeqModel 객체를 이용해서 모델을 만들거나(train), 사용(decoder)하는 부분으로 컨트롤할 계획.
import seq2seq_model

"""
내부적으로 seq2seq 모델을 사용하는 BE용 모듈, 
create_model은 생성할때 만들어진 model이 있는지 판단하여 내부적으로 그래프를 생성한다.
train() 은 훈련데이터를 입력하면 별도로 Seq2SeqModel을 만들고 그것으로 훈련을 시작한다.
decode() 는 입력 데이터를 주면 그것으로 추론한 데이터를 반환한다. (service)
"""
buckets = [(5, 5), (10, 10), (15, 15), (30, 30), (40, 40)]
model_dir ="./seq2seq_bemodel"

hidden_size = 150
num_layers = 3
max_gradient_norm = 5.0
batch_size = 100
learning_rate = 0.0001
learning_rate_decay_factor = 0.99
use_lstm = True
steps_per_checkpoint = 200

def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""

    enc_dict, dec_dict = load_vocab("./final/Encoder_Dict_key2val_file_forBE", "./final/Decoder_Dict_val2key_file_forBE")

    model = seq2seq_model.Seq2SeqModel(
        source_vocab_size= len(enc_dict),
        target_vocab_size= len(dec_dict),
        buckets=buckets,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        max_gradient_norm=max_gradient_norm,
        learning_rate=learning_rate,
        learning_rate_decay_factor=learning_rate_decay_factor,
        isbe=True,
        use_lstm=use_lstm,
        forward_only=forward_only,
        dtype=tf.float32)  # 객체가 내부적으로 유지하는 우리의 seq2seq_model

    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s"% ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)  # 이미 checkpoint 존재하면 로드
    else:
        print("Create model with new parameters.")
        session.run(tf.global_variables_initializer())  # 없으면 그래프 초기화.

    return model

def train():
    with tf.Session() as sess:
        # 데이터는 학습할 때만 사실 학습할 때만 쓰겠지.(테스트도 가능하겠으나..)
        #Create model.
        print("Creating {} layers of {} units.".format(num_layers, hidden_size))
        # model = create_model(sess, False) # 기존
        with tf.variable_scope("be_model"):
            model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        enc_inputs,dec_inputs,enc_vocab,rev_dec_vocab,_,_,_,_=load_data_v2("./final/data_recent.csv",
                                                                           "./final/Encoder_Dict_key2val_file_forBE",
                                                                           "./final/Decoder_Dict_key2val_file_forBE",
                                                                           "./final/Decoder_Dict_val2key_file_forBE",
                                                                           "./final/Encoder_Dict_key2val_file_forGG",
                                                                           "./final/Decoder_Dict_key2val_file_forGG",
                                                                           "./final/Decoder_Dict_val2key_file_forGG")
        train_set, test_set = model.read_data(enc_inputs, dec_inputs)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i+1]) / train_total_size for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample() # returns float (0, -1)
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])


            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id) # get batch_major datas, (inputs length*batch_size)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
            step_time += (time.time() - start_time)/steps_per_checkpoint # 평가 단계에서의 평균이네..
            loss += step_loss / steps_per_checkpoint # 평가 단계에서의 평균..

            current_step +=1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % steps_per_checkpoint == 0:
                #Print statistics for the previous epoch.

                perplexity = math.exp(float(loss)) if loss < 300 else float("inf") # 일정치 이상 loss가 높으면 무한이라 판단.
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_name = "/seq2seq.ckpt"
                # Save checkpoint and zero timer and loss.
                model.saver.save(sess, model_dir+checkpoint_name, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(buckets)):
                    if len(test_set[bucket_id]) == 0:
                        print(" eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print(" eval:bucket %d perplexity %.2f" % (bucket_id, eval_ppx))


def decode(sess, model, query):
    # Load vocabularies.
    enc_vocab, rev_dec_vocab = load_vocab("./final/Encoder_Dict_key2val_file_forBE", "./final/Decoder_Dict_val2key_file_forBE")

    # Get token-ids for the input sentence.
    tokens = utils.query_disintegrator(query)
    token_ids = utils.sent2idx(tokens, enc_vocab)

    # Which bucket does it belong to?
    bucket_id = len(buckets) -1
    for i, bucket in enumerate(buckets):
        if bucket[0] >= len(token_ids):
            bucket_id = i
            break

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id) #getbatch 적용하기위한 load_data수정필요

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True) # output_logits, (decoder_length*batch_size*dec_vocab_size)

    #output_logits = np.array(output_logits)
    #print("debug", output_logits.shape)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.

    #outputs = [int(np.argmax(logits, axis=1)) for logits in output_logits]
    outputs = [np.argmax(logits, axis=-1) for logits in output_logits]
    outputs = np.transpose(outputs)
    output = list(outputs[0])

    # If there is an EOS symbol in outputs, cut them at that point.
    EOS_ID = 2 # dec_vocab["<EOS>"]
    if EOS_ID in output:
        output = output[:output.index(EOS_ID)]
    else:
        sent = utils.tokenize("할 말이 없다")
        output = utils.sent2idx(sent, enc_vocab)

    # print out decoder sentence corresponding to outputs
    #answer = " ".join([rev_dec_vocab[str(token)] for token in output])
    answer = [rev_dec_vocab[str(token)] for token in output]

    return answer




