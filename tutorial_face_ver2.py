import tensorflow as tf
import numpy as np

'''
network configuration:
-input 96x96x3
-conv1 96x96x64
-conv2 48x48x128
-conv3 24x24x256
-conv4 12x12x512
-conv5 6x6x512
-fc6 3x3x512x4096
-fc7 4096x4096
-fc8 4096x3
'''

class HyperParameters:
    def __init__(self):
        # for training
        self.x_train = np.load("saving/npy/ver1/x_train1.npy")
        self.train_label = np.load("saving/npy/ver1/x_train_label1.npy")

        self.x_test = np.load("saving/npy/ver1/x_test1.npy")
        self.test_label = np.load("saving/npy/ver1/x_test_label1.npy")

        self.save_path = 'backup/icpt_tutorial'

        self.learning_rate = 0.001
        self.keep_prob = tf.placeholder(tf.float32)

        # for build network
        self.value = 3
        self.kernel_size_dict = {'conv1': [3, 3, 3, 64],
                                 'conv2': [3, 3, 64, 128],
                                 'conv3': [3, 3, 128, 256],
                                 'conv4': [3, 3, 256, 512],
                                 'conv5': [3, 3, 512, 512],
                                 'fc6': [3 * 3 * 512, 4096],
                                 'fc7': [4096, 4096],
                                 'fc8': [4096, self.value]}
        self.weight_dict = {}
        for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
            self.weight_dict[name] = tf.Variable(tf.random_normal(self.kernel_size_dict[name], stddev=0.01))


class Network:
    def __init__(self):
        self.info = HyperParameters()
        self.weight_dict = self.info.weight_dict

    def max_pool(self, layer):
        return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def conv_layer(self, layer, name):
        weight = self.weight_dict[name]
        return tf.nn.relu(tf.nn.conv2d(layer, weight, strides=[1, 1, 1, 1], padding='SAME'))

    def fc_layer(self, layer, name):
        weight = self.weight_dict[name]
        return tf.nn.relu(tf.matmul(layer, weight))

    def build(self):  # input: _, 96, 96, 3
        # conv_layer
        self.X = tf.placeholder(tf.float32, [None, 96, 96, 3])
        self.Y = tf.placeholder(tf.float32, [None, self.info.value])

        self.conv1 = self.conv_layer(self.X, 'conv1')
        self.conv1 = self.max_pool(self.conv1)
        self.conv2 = self.conv_layer(self.conv1, 'conv2')
        self.conv2 = self.max_pool(self.conv2)
        self.conv3 = self.conv_layer(self.conv2, 'conv3')
        self.conv3 = self.max_pool(self.conv3)
        self.conv4 = self.conv_layer(self.conv3, 'conv4')
        self.conv4 = self.max_pool(self.conv4)
        self.conv5 = self.conv_layer(self.conv4, 'conv5')
        self.conv5 = self.max_pool(self.conv5)

        # fc_layer
        self.conv5 = tf.reshape(self.conv5, [-1, 3 * 3 * 512])
        self.fc6 = self.fc_layer(self.conv5, 'fc6')
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
        self.model = self.fc_layer(self.fc7, 'fc8')

        sess2 = tf.Session()
        sess2.run(tf.global_variables_initializer())

        return self.model


class Training:
    def __init__(self):
        info = HyperParameters()

        self.x_train = info.x_train
        self.train_label = info.train_label

        self.x_test = info.x_test
        self.test_label = info.test_label

        self.learning_rate = info.learning_rate
        self.keep_prob = info.keep_prob

        self.save_path = info.save_path

        init = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        sess.run(init2)
        self.sess = sess

        self.saving(info.save_path)

    def saving(self, save_path):
        if tf.train.get_checkpoint_state(save_path):
            saver = tf.train.Saver()
            saver.restore(self.sess, save_path)

    def train(self, ranges):
        net = Network()
        model = net.build()

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=net.Y, logits=model))
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        batch_size = 100
        total_batch = int(self.x_train.shape[0] / batch_size)

        for epoch in range(ranges):
            total_cost = 0
            for i in range(total_batch):
                _, cost_val = self.sess.run([optimizer, cost],
                                            feed_dict={net.X: self.x_train,
                                                       net.Y: self.train_label,
                                                       self.keep_prob: 0.7})

                total_cost += cost_val
                saver = tf.train.Saver()
                saver.save(self.sess, self.save_path, write_meta_graph=False)
            print('Epoch:{}'.format(epoch + 1),
                  'cost=', '{:3f}'.format(total_cost))

        print('finish')

    def test(self):
        net = Network()
        model = net.build()

        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(net.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        acc = self.sess.run(accuracy,
                            feed_dict={net.X: self.x_test,
                                       net.Y: self.test_label,
                                       self.keep_prob: 1})
        print('accuracy:', acc)


"""
class Using_Network:
    def __init__(self):
        self.params = HyperParameters()
        self.model = Network().build()
        Training().saving(self.params.save_path)
    def load_image(self):

    def image_input(self):
        results = tf.nn.softmax(self.model)

"""


def main():
    a = Training()
    a.train(1)
    a.test()


if __name__ == '__main__':
    main()
