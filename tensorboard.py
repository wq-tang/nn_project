import os
import numpy as np 
import tensorflow as tf 
import time
import cifar10_input
import math
from comparable_model import complex_net
from tensorflow.examples.tutorials.mnist import input_data
##cifar batch =128  epoch = 50000
##mnist epoch=50  bathch = 60000
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def mnist():
    def test(images,labels,accuracy):
        p = 0
        for i in range(10):
            xs = images[i*1000:(i+1)*1000]
            ys = labels[i*1000:(i+1)*1000]
            p+= accuracy.eval(feed_dict={x:xs, y:ys})
        return p/10
    mnist_data_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'mnist') 
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'tensorboard')
    mnist=input_data.read_data_sets(mnist_data_folder,one_hot=True)
    epoch = 50
    batch = 100
    x  = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.int32,[None,10])
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)
    model = [complex_net(image_shaped_input,10,0)]
    models_result =model[0].out
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(models_result, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=models_result))
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.1**3).minimize(cross_entropy) 
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    ans = []
    for i in range(epoch*10):
        start_time = time.time()
        train_x, train_y = mnist.train.next_batch(batch)
        _ = sess.run( train_step, feed_dict={x:train_x,y:train_y})
        duration = time.time() - start_time
        if i%100 ==0:
            examples_per_sec = batch/duration
            sec_per_batch = float(duration)
            format_str = ('step %d,loss=%.2f (%.1f examples/sec; %.3f sec/batch)')
            summary,loss_value = sess.run([merged,cross_entropy], feed_dict={x:train_x,y:train_y})
            train_writer.add_summary(summary, i)
            train_accuracy = accuracy.eval(feed_dict={x:train_x, y:train_y})
            print(format_str %(i,loss_value,examples_per_sec,sec_per_batch))
            for m in model:
                m.training = False
            summary, acc = sess.run([merged, accuracy], feed_dict={x:mnist.test.images,y:mnist.test.labels})
            test_writer.add_summary(summary, i)
            test_accuracy = test(mnist.test.images,mnist.test.labels,accuracy)
            ans.append(test_accuracy)
            for m in model:
                m.training = True
            print( "step %d, training accuracy %g"%(i, train_accuracy))
            print( "step %d,test accuracy %g"%(i,test_accuracy))
    train_writer.close()
    test_writer.close()
    print('precision @1 = %.5f'%np.mean(ans[-100:]))
if __name__=='__main__':
	mnist()
