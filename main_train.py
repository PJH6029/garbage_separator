import tensorflow as tf
import numpy as np
import os

# 인공신경망 구조
conv_num = 3  # Convolutional Layer 개수
conv1_size = 32
conv2_size = 64
conv3_size = 128
conv4_size = 64
conv5_size = 64  # Convolutional Layer 크기(깊이)
fc6_size = 128
fc7_size = 256  # Fully Connected Layer 크기
learning_rate = 0.0005
save_path = {3: 'backup/better_model/{0},{1},{2},{3},{4},{5}/'.format(conv1_size, conv2_size, conv3_size,
                                                                      learning_rate, fc6_size, fc7_size),
             4: 'backup/better_model/{0},{1},{2},{3},{4},{5},{6}/'.format(conv1_size, conv2_size, conv3_size,
                                                                          conv4_size, learning_rate, fc6_size,
                                                                          fc7_size),
             5: 'backup/better_model/{0},{1},{2},{3},{4},{5},{6},{7}/'.format(conv1_size, conv2_size, conv3_size,
                                                                              conv4_size, conv5_size, learning_rate,
                                                                              fc6_size, fc7_size)}[conv_num]
accuracy_save_path = 'experiment_data/accuracy.txt'
cost_save_path = 'experiment_data/cost.txt'
both_save_path = 'experiment_data/accuracy,cost.txt'
if not (os.path.isdir(save_path[:-1])):
    os.makedirs(save_path[:-1])
drop_out = 0.7  # Drop_Out 비율
value = 16  # 최종 클래스 개수(알로에~요구르트)
img_size = 96  # 이미지 크기
dim = {3: int(img_size / 8) * int(img_size / 8) * conv3_size,
       4: int(img_size / 16) * int(img_size / 16) * conv4_size,
       5: int(img_size / 32) * int(img_size / 32) * conv5_size}[conv_num]
kernel_size_dict = {'conv1': [3, 3, 3, conv1_size],
                    'conv2': [3, 3, conv1_size, conv2_size],
                    'conv3': [3, 3, conv2_size, conv3_size],
                    'conv4': [3, 3, conv3_size, conv4_size],
                    'conv5': [3, 3, conv4_size, conv5_size],
                    'fc6': [dim, fc6_size],
                    'fc7': [fc6_size, fc7_size],
                    'fc8': [fc7_size, value]}
keep_prob = tf.placeholder(tf.float32)
epoch_range = 30  # 학습 횟수
weight_dict = {}
for name in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
    weight_dict[name] = tf.Variable(tf.truncated_normal(kernel_size_dict[name], stddev=0.01))


# Pooling(중요한 부분 추출) 함수
def max_pool(layer):
    return tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Convolutional Layer 함수
def conv_layer(layer, name):
    weight = weight_dict[name]
    return tf.nn.relu(tf.nn.conv2d(layer, weight, strides=[1, 1, 1, 1], padding='SAME'))


# Fully Connected Layer 함수
def fc_layer(layer, name):
    weight = weight_dict[name]
    return tf.nn.relu(tf.matmul(layer, weight))


# build network
X = tf.placeholder(tf.float32, [None, img_size, img_size, 3])
Y = tf.placeholder(tf.float32, [None, value])
# conv_layer
conv1 = conv_layer(X, 'conv1')
conv1 = max_pool(conv1)
conv2 = conv_layer(conv1, 'conv2')
conv2 = max_pool(conv2)
conv3 = conv_layer(conv2, 'conv3')
conv3 = max_pool(conv3)
conv4 = conv_layer(conv3, 'conv4')
conv4 = max_pool(conv4)
conv5 = conv_layer(conv4, 'conv5')
conv5 = max_pool(conv5)
# fc_layer
flat = {3: tf.reshape(conv3, [-1, dim]),
        4: tf.reshape(conv4, [-1, dim]),
        5: tf.reshape(conv5, [-1, dim])}[conv_num]
fc6 = fc_layer(flat, 'fc6')
fc7 = fc_layer(fc6, 'fc7')
fc7 = tf.nn.dropout(fc7, keep_prob)
model = fc_layer(fc7, 'fc8')
model = tf.nn.softmax(model)
print(model.shape)
# 오차함수 생성 및 경사하강법을 이용한 최적하 함수 생성
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# ---------------------------------------------------------------------------------
# 인공신경망 모델 학습
if tf.train.get_checkpoint_state(save_path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path)
    print('model loaded')
print('model:' + save_path[24:-1])
config_writer = open(save_path + 'config.txt', 'w')
config = 'learning_rate={0}\nvalue={1}\nimg_size={2}\ndim={3}\nkernel_size_dict={4}'.format(
    learning_rate, value, img_size, dim, kernel_size_dict
)
config_writer.write(config)
config_writer.close()
train_batch = 28
total_cost_list = []
for epoch in range(epoch_range):
    total_cost = 0
    for i in range(train_batch):
        x_train = np.load('saving/npy/cut_100/x_train{}.npy'.format(i))
        train_label = np.load('saving/npy/cut_100/x_train_label{}.npy'.format(i))
        if epoch == 0:
            print(x_train.shape)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: x_train,
                                          Y: train_label,
                                          keep_prob: drop_out})
        total_cost += cost_val
        saver = tf.train.Saver()
        saver.save(sess, save_path, write_meta_graph=False)
    total_cost_list.append(total_cost)
    print('Epoch:{0}'.format(epoch + 1),
          'cost=', '{:3f}'.format(total_cost))
for i in range(len(total_cost_list)):
    total_cost_list[i] = '{:3f}'.format(total_cost_list[i])
cost_list_for_write = " ".join(total_cost_list)
cost_writer = open(save_path + 'cost_val.txt', 'w')
cost_writer.write(cost_list_for_write)
cost_writer.close()
cost_txt_writer = open(cost_save_path, 'a')
cost_txt_writer.write(save_path[24:-1] + ':\t')
cost_txt_writer.write(cost_list_for_write)
cost_txt_writer.write('\n')
cost_txt_writer.close()
print('finish')
# ----------------------------------------------------------------------------
# 인공신경망 모델 테스트
sum_acc = 0
test_batch = 12
total_acc = 0
for i in range(test_batch):
    x_test = np.load('saving/npy/cut_100/x_test{}.npy'.format(i))
    test_label = np.load('saving/npy/cut_100/x_test_label{}.npy'.format(i))
    print(x_test.shape)
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    acc = sess.run(accuracy,
                   feed_dict={X: x_test,
                              Y: test_label,
                              keep_prob: 1})
    sum_acc += acc
    if i == (test_batch - 1):
        accuracy_writer = open(save_path + 'accuracy.txt', 'w')
        accuracy_txt_writer = open(accuracy_save_path, 'a')
        total_acc = sum_acc / test_batch
        print('accuracy:', total_acc)
        accuracy_writer.write(str(total_acc))
        accuracy_txt_writer.write(save_path[24:-1] + ':\t' + str(total_acc) + '\n')
        accuracy_writer.close()
        accuracy_txt_writer.close()
both_writer = open(both_save_path, 'a')
content = save_path[24:-1] + ':\t' + str(total_acc) + ',\t' + cost_list_for_write + '\n'
both_writer.write(content)
both_writer.close()
