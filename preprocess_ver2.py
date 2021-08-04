import os
import glob
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

image_path = glob.glob('saving/trash/*')
train_raw_path = 'saving/raw/trash_96x96/train/'
test_raw_path = 'saving/raw/trash_96x96/test/'
train_npy_path = 'saving/npy/trash_96x96/x_train{}.npy'
train_label_npy_path = 'saving/npy/trash_96x96/x_train_label{}.npy'
test_npy_path = 'saving/npy/trash_96x96/x_test{}.npy'
test_label_npy_path = 'saving/npy/trash_96x96/x_test_label{}.npy'

img_size = 96
value = 3

def preprocess(image_path, train_raw_path, test_raw_path,
               train_path, train_label_path, test_path, test_label_path):
    print('... loading data')

    data_path = []
    for path in image_path:
        trashes = glob.glob(path)
        trashes_list = np.array(
            [e for x in [glob.glob(os.path.join(trash, '*'))
            for trash in trashes] for e in x]
        )
        name = os.path.basename(path)
        name_list = [name for i in range(len(trashes_list))]
        data_path += list(zip(trashes_list, name_list))

    np.random.shuffle(data_path)

    label = []
    data_list = []
    for picture_path in data_path:
        img = cv2.imread(picture_path[0])
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        data_list.append(img)
        if img is None:
            continue
        label.append(picture_path[1])

    #integer encode
    label = np.array(label)
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(label)

    #onehot encode
    onehot_encoder = OneHotEncoder(n_values=value)
    label = onehot_encoder.fit_transform(label.reshape(-1, 1)).toarray()

    data = [[data_list[i], label[i]] for i in range(len(label))]

    proportion = int(len(data) * 0.7)
    train_part = data[:proportion]
    test_part = data[proportion:]

    #train
    train_record = open(train_raw_path+'train_label.txt', 'w')
    test_record = open(test_raw_path+'test_label.txt', 'w')
    for i, p in enumerate(train_part):
        name = '{}.jpg'.format('{0:05d}'.format(i))
        imgpath = os.path.join(train_raw_path, name)
        cv2.imwrite(imgpath, p[0])
        train_record.write(str(p[1]) + '\n')

    for i, p in enumerate(test_part):
        name = '{}.jpg'.format('{0:05d}'.format(i))
        imgpath = os.path.join(test_raw_path, name)
        cv2.imwrite(imgpath, p[0])
        test_record.write(str(p[1])+'\n')
    train_record.close()
    test_record.close()

    #cutter = int(len(train_part) * 0.1)
    cutter = 300
    for i in range(int(len(train_part)/cutter)+1):
        train_part_cut = train_part[cutter * i : cutter * int(i + 1)]
        x_train = np.array([img[0] for img in train_part_cut])
        x_train_label = np.array([label[1] for label in train_part_cut])
        np.save(train_path.format(i), x_train)
        np.save(train_label_path.format(i), x_train_label)

        print('x_train{}:'.format(i), x_train.shape)
        print('x_train_label{}:'.format(i), x_train_label.shape)

    for i in range(int(len(test_part)/cutter)+1):
        test_part_cut = test_part[cutter * i : cutter * int(i+1)]
        x_test = np.array([img[0] for img in test_part_cut])
        x_test_label = np.array([label[1] for label in test_part_cut])
        np.save(test_path.format(i), x_test)
        np.save(test_label_path.format(i), x_test_label)

        print('x_test{}:'.format(i), x_test.shape)
        print('x_test_label{}:'.format(i), x_test_label.shape)




def main():
    preprocess(image_path, train_raw_path, test_raw_path,
               train_npy_path, train_label_npy_path, test_npy_path, test_label_npy_path)

if __name__ == '__main__':
    main()