import cv2
import glob
import os
import numpy as np
import tensorflow as tf

save_path = 'saving/live/'
model_path = 'backup/icpt_recycling/trash_split_class/'

def trash_capture():
    global picture
    picture = []
    vidcap = cv2.VideoCapture(0)
    vidcap.set(3, 320)
    vidcap.set(4, 240)

    while vidcap.isOpened():

        sucess, img = vidcap.read()
        if not sucess:
            break

        cv2.imshow('webcam', img)
        if(cv2.waitKey(1)&0xFF == ord('q')):
            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
            picture.append(img[:])
            cv2.waitKey(2000)
            break

    picture = np.array(picture)


def test():
    keep_prob = tf.placeholder(tf.float32)

    value = 16
    img_size = 96

    dim = int(img_size / 16) * int(img_size / 16) * 64

    kernel_size_dict = {'conv1': [3, 3, 3, 32],
                        'conv2': [3, 3, 32, 64],
                        'conv3': [3, 3, 64, 128],
                        'conv4': [3, 3, 128, 64],
                        'conv5': [3, 3, 64, 64],
                        'fc6': [dim, 256],
                        'fc7': [256, 128],
                        'fc8': [128, value]}

    weight_dict = {}
    for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
        weight_dict[name] = tf.Variable(tf.random_normal(kernel_size_dict[name], stddev=0.01))

    def max_pool(layer):
        return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def conv_layer(layer, name):
        weight = weight_dict[name]
        return tf.nn.relu(tf.nn.conv2d(layer, weight, strides=[1, 1, 1, 1], padding='SAME'))

    def fc_layer(layer, name):
        weight = weight_dict[name]
        return tf.nn.relu(tf.matmul(layer, weight))

    # bulid network

    X = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
    Y = tf.placeholder(tf.float32, [None, value])

    conv1 = conv_layer(X, 'conv1')
    conv1 = max_pool(conv1)
    conv2 = conv_layer(conv1, 'conv2')
    conv2 = max_pool(conv2)
    conv3 = conv_layer(conv2, 'conv3')
    conv3 = max_pool(conv3)
    conv4 = conv_layer(conv3, 'conv4')
    conv4 = max_pool(conv4)
    # print(conv4.shape)
    # conv4 = conv_layer(conv3, 'conv4')
    # conv4 = max_pool(conv4)
    conv5 = conv_layer(conv4, 'conv5')
    conv5 = max_pool(conv5)

    # fc_layer
    conv4 = tf.reshape(conv4, [-1, dim])
    fc6 = fc_layer(conv4, 'fc6')
    fc7 = fc_layer(fc6, 'fc7')
    fc7 = tf.nn.dropout(fc7, keep_prob)
    model = fc_layer(fc7, 'fc8')
    model = tf.nn.softmax(model)
    print(model.shape)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    if tf.train.get_checkpoint_state(model_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print('model loaded')


    '''
    predict = tf.argmax(model, 1)
    trash = sess.run(predict, feed_dict={X:picture,
                                         keep_prob: 1.0})

    if trash == 0:
        print('캔')
    elif trash == 1:
        print('페트')
    '''
    print(sess.run(model, feed_dict={X:picture, keep_prob:1.0}))




def main():
    trash_capture()
    test()

if __name__ == '__main__':
    main()