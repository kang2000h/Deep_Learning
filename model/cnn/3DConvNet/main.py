import tensorflow as tf


from model import *
from utils import *
from dicom_loader import dicom_loader
from model import Conv3DNet
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, help="dicom directory you need to show")
# parser.add_argument('--output_path', type=str,
#                     help="dir of numpy array that have dicoms pixels")
# parser.add_argument('--save_filename', type=str,
#                     help="filename of numpy array that have dicoms pixels")
# parser.add_argument('--visualize', type=bool,
#                     help="if you need to visualize")
# args = parser.parse_args()

flags = tf.app.flags
flags.DEFINE_string("mode", None, "define what to do")
flags.DEFINE_string("data_path", "./data/muchdata-50-50-20.npy", "data_path to use")
flags.DEFINE_integer("batch_size", 20, "batch_size of model")
flags.DEFINE_integer("f_filter", 32, "number of first filter")
flags.DEFINE_float("beta1", 0.5, "momentum term of adam")
flags.DEFINE_string("train_type", "tvt", "whether to do test on training")
flags.DEFINE_float("train_rate", 0.7, "how many take the samples as a training data")
flags.DEFINE_float("learning_rate", 0.0001, "how many take the samples as a training data")
flags.DEFINE_integer("epoch", 50, "epoch to train")
flags.DEFINE_string("model_dir", None, "model_dir to load")

FLAGS = flags.FLAGS
def main(_):
    pp.pprint(flags.FLAGS.__flags)

    much_data = dicom_loader(FLAGS.data_path)  # num_classes, num_data, depth, height, width -> not good, we need to be robust on even unbalanced classes

    if much_data.shape[-1] != 3 and len(much_data.shape)==5:
        much_data = np.expand_dims(much_data, axis=len(much_data.shape))

        print("Data loaded completely", much_data.shape)
        num_classes = much_data.shape[0]
        all_data_size = much_data.shape[1]
        input_depth = much_data.shape[2]
        input_height = much_data.shape[3]
        input_width = much_data.shape[4]
        input_cdim = much_data.shape[5]

    elif much_data.shape[-1] != 1 and np.array(much_data[0]).shape[-1] !=3 : # to overcome unbalanced classes data, e.g num_classes,
        print("data seem to be unbalanced class data")
        for ind in range(len(much_data)):
            much_data[ind] = np.expand_dims(much_data[ind], axis=len(np.array(much_data[ind]).shape))
        much_data = np.array(much_data)
        print("Data loaded completely", much_data.shape)
        num_classes = much_data.shape[0]
        all_data_size = much_data[0].shape[0]
        input_depth = much_data[0].shape[1]
        input_height = much_data[0].shape[2]
        input_width = much_data[0].shape[3]
        input_cdim = much_data[0].shape[4]




    batch_size = FLAGS.batch_size
    f_filter= FLAGS.f_filter

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:


        if FLAGS.mode == 'train':

            forward_only = False
            model = Conv3DNet(sess, input_depth, input_height, input_width, input_cdim, num_classes, all_data_size, FLAGS.batch_size, FLAGS.epoch, FLAGS.model_dir,
                              FLAGS.learning_rate, FLAGS.train_rate, FLAGS.train_type,
                              FLAGS.f_filter, FLAGS.beta1, forward_only)

            model.create_model() # Structuring proper tensor graph


            if FLAGS.train_type == "tvt":
                tr_data, val_data, test_data ,\
                    tr_label, val_label, test_label = model.get_batch(much_data)
                x = (tr_data, val_data, test_data)
                y = (tr_label, val_label, test_label)
                model.train(x, y)

            elif FLAGS.train_type == "tv":
                tr_data, val_data,  \
                tr_label, val_label = model.get_batch(much_data)


        elif FLAGS.mode=='test':
            forward_only = True
            model = Conv3DNet(sess, input_depth, input_height, input_width, input_cdim, num_classes, all_data_size,
                              FLAGS.batch_size,
                              FLAGS.train_rate, FLAGS.train_type,
                              f_filter, FLAGS.beta1, forward_only)
            #model.test()

if __name__=="__main__":
    tf.app.run()