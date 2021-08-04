import tensorflow as tf
import numpy as np

#96x96: 300장이 최대

x_train = np.load('saving/npy/how_many_96x96/x_train0.npy')
train_label = np.load('saving/npy/how_many_96x96/x_train_label0.npy')
x_test = np.load('saving/npy/how_many_96x96/x_test.npy')
test_label = np.load('saving/npy/how_many_96x96/x_test_label.npy')

print(x_train.shape)

save_path = 'backup/how_many_96x96/'

learning_rate = 0.001
keep_prob = tf.placeholder(tf.float32)

value=3
epoch_range = 10

img_size = 96

dim = int(img_size/16) * int(img_size/16) * 64

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

#bulid network

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
#print(conv4.shape)
#conv4 = conv_layer(conv3, 'conv4')
#conv4 = max_pool(conv4)
conv5 = conv_layer(conv4, 'conv5')
conv5 = max_pool(conv5)

# fc_layer
conv4 = tf.reshape(conv4, [-1, dim])
fc6 = fc_layer(conv4, 'fc6')
fc7 = fc_layer(fc6, 'fc7')
model = fc_layer(fc7, 'fc8')
print(model.shape)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer() #변수들을 모두 시작. optimizer 뒤에 나와야함
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
sess = tf.Session()
sess.run(init)

if tf.train.get_checkpoint_state(save_path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path)
    print('model loaded')

batch_size = 100
#total_batch = int(x_train.shape[0] / batch_size)
total_batch = 1
for epoch in range(epoch_range):
    total_cost = 0
    for i in range(total_batch):
        _, cost_val = sess.run([optimizer, cost],
                                    feed_dict={X:x_train,
                                               Y:train_label,
                                               keep_prob: 1})
        total_cost += cost_val
        saver = tf.train.Saver()
        saver.save(sess, save_path, write_meta_graph=False)
    print('Epoch:{0}'.format(epoch+1),
          'cost=', '{:3f}'.format(total_cost))

print('finish')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
acc = sess.run(accuracy,
               feed_dict={X:x_test,
                          Y:test_label,
                          keep_prob:1})
print('accuracy:', acc)
