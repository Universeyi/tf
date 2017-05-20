import time
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
import csv
EVAL_INTERVAL_SECS = 10
data_path_samples = "test.csv"
TEST_RESULT_PATH = "output.csv"
def evaluate(X_matirx,sample_num):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        # y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: X_matirx}

        y = mnist_inference.inference(x, None)
        predict_num = tf.argmax(y, 1)
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)


        with tf.Session() as sess:
            csvfile = file(TEST_RESULT_PATH,"wb")
            writer = csv.writer(csvfile)
            writer.writerow(['ImageId','Label'])
            index_col = [i for i in range(1,sample_num+1)]
            # index_col_array = np.asarray(index_col)
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                #accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                sess.run(predict_num,feed_dict=validate_feed)
                data = [index_col,predict_num.tolist()]
                writer.writerow(data)
                csvfile.close()
                # print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return


def main(argv=None):
    # mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    X_matirx = pd.read_csv(filepath_or_buffer=data_path_samples,delim_whitespace=True,header=1).as_matrix()
    sample_num = X_matirx.shape[0]
    evaluate(X_matirx,sample_num)
if __name__ == '__main__':
    main()
