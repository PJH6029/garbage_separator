import tensorflow as tf
import numpy as np

#96x96: 300장이 최대

#0:알로에 1:사과 2:바나나 3:블루 4:코코팜 5:자몽 6:제티 7:레몬 8:미란다 9:모히또 10:매실 11:파워캔 12:파워페트 13:삼다수 14:석류 15:요구르트

save_path = 'backup/model_experiment/32,64,128,0.001,128,128/'

learning_rate = 0.001
drop_out = 0.7
value=16

img_size = 96

dim = int(img_size/8) * int(img_size/8) * 128

kernel_size_dict = {'conv1': [3, 3, 3, 32],
'conv2': [3, 3, 32, 64],
'conv3': [3, 3, 64, 128],
'conv4': [3, 3, 128, 64],
'conv5': [3, 3, 64, 64],
'fc6': [dim, 128],
'fc7': [128, 128],
'fc8': [128, value]}

keep_prob = tf.placeholder(tf.float32)
epoch_range = 30


weight_dict = {}
for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
    weight_dict[name] = tf.Variable(tf.truncated_normal(kernel_size_dict[name], stddev=0.01))


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
flat = tf.reshape(conv3, [-1, dim])
fc6 = fc_layer(flat, 'fc6')
fc7 = fc_layer(fc6, 'fc7')
fc7 = tf.nn.dropout(fc7, keep_prob)
model = fc_layer(fc7, 'fc8')
model = tf.nn.softmax(model)
print(model.shape)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer() #변수들을 모두 시작. optimizer 뒤에 나와야함
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
sess = tf.Session()
sess.run(init)


#---------------------------------------------------------------------------------


if tf.train.get_checkpoint_state(save_path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path)
    print('model loaded')

config_writer = open(save_path+'config.txt', 'w')
config = 'learning_rate={0}\nvalue={1}\nimg_size={2}\ndim={3}\nkernel_size_dict={4}'.format(
    learning_rate, value, img_size, dim, kernel_size_dict
)
config_writer.write(config)
config_writer.close()

train_batch = 28
for epoch in range(epoch_range):
    total_cost = 0
    for i in range(train_batch):

        x_train = np.load('saving/npy/cut_100/x_train{}.npy'.format(i))
        train_label = np.load('saving/npy/cut_100/x_train_label{}.npy'.format(i))

        if epoch == 0:
            print(x_train.shape)

        _, cost_val = sess.run([optimizer, cost],
                                feed_dict={X:x_train,
                                            Y:train_label,
                                            keep_prob: drop_out})
        total_cost += cost_val
        saver = tf.train.Saver()
        saver.save(sess, save_path, write_meta_graph=False)


    cost_writer = open(save_path+'cost_val.txt', 'a')
    print('Epoch:{0}'.format(epoch+1),
            'cost=', '{:3f}'.format(total_cost))
    cost_writer.write('{:3f}'.format(total_cost)+'\n')
    cost_writer.close()

print('finish')


#----------------------------------------------------------------------------

sum_acc = 0
test_batch = 12
for i in range(test_batch):
    x_test = np.load('saving/npy/cut_100/x_test{}.npy'.format(i))
    test_label = np.load('saving/npy/cut_100/x_test_label{}.npy'.format(i))

    print(x_test.shape)


    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    acc = sess.run(accuracy,
               feed_dict={X:x_test,
                          Y:test_label,
                          keep_prob:1})

    sum_acc += acc

    if i == (test_batch-1):
        accuracy_writer = open(save_path+'accuracy.txt', 'w')
        total_acc = sum_acc/(test_batch)
        print('accuracy:', total_acc)
        accuracy_writer.write(total_acc)
        accuracy_writer.close()
        '''
    if i == 3:
        print(sess.run(model, feed_dict={X: x_test, keep_prob: 1.0}))
'''