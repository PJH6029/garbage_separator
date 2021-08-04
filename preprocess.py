import os
import glob
import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#from scipy import sparse

#member_list = [['jeonghun', 1], ['gwanbin', 2]]

image_path = 'saving/raw_face/'
train_path = 'saving/raw/how_many_96x96/train/'
test_path = 'saving/raw/how_many_96x96/test/'
train_save_path = 'saving/npy/how_many_96x96/x_train{}.npy'
train_label_save_path = 'saving/npy/how_many_96x96/x_train_label{}.npy'
test_save_path = 'saving/npy/how_many_96x96/x_test.npy'
test_label_save_path = 'saving/npy/how_many_96x96/x_test_label.npy'

image_size = 96

def main():
    preprocess(image_path, train_path, test_path,
               train_save_path, train_label_save_path, test_save_path, test_label_save_path)


def preprocess(image_path, train_path, test_path,
               train_save_path, train_label_save_path, test_save_path, test_label_save_path):
    print('... loading data')
    #if 디렉토리 없으면
    #os.mkdir('saving/raw/train1')
    #os.mkdir('saving/raw/test1')

    persons = glob.glob(image_path)
    paths = np.array(
        [e for x in [glob.glob(os.path.join(person, '*'))
        for person in persons] for e in x])
    np.random.shuffle(paths)

    #r = int(len(paths) * 0.40)   #데이터를 train과 test로 분류
    #a = int(len(paths) * 0.60)
    r = int(len(paths) * 0.7)
    train_part = paths[:r]
    test_part = paths[r:]

    x_train = []
    x_train_label = []
    for i, d in enumerate(train_part):
        img = cv2.imread(d)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA) #placeholder에 맞게 resize
        if img is None:
            continue
        x_train.append(img)
        name = "{}.jpg".format("{0:05d}".format(i))
        imgpath = os.path.join(train_path, name)
        cv2.imwrite(imgpath, img)
    for path in train_part:
        name = os.path.basename(path)
        name_first=name[0]
        x_train_label.append(name_first)


    x_test = []
    x_test_label = []
    for i, d in enumerate(test_part):
        img = cv2.imread(d)
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        if img is None:
            continue
        x_test.append(img)
        name = "{}.jpg".format("{0:05d}".format(i))
        imgpath = os.path.join(test_path, name)
        cv2.imwrite(imgpath, img)
    for path in test_part:
        name = os.path.basename(path)
        name_first=name[0]
        x_test_label.append(name_first)


    #print(x_train_label)
    #print(x_test_label)
    x_train_label = np.array(x_train_label)
    x_test_label = np.array(x_test_label)
    onehot_encoder = OneHotEncoder(sparse=False)
    x_train_label, x_test_label = x_train_label.reshape(len(x_train_label), 1), x_test_label.reshape(len(x_test_label),1)
    x_train_label, x_test_label = onehot_encoder.fit_transform(x_train_label), onehot_encoder.fit_transform(x_test_label)
    #print(x_train_label)

    '''
    for_saving=[x_train, x_train_label, x_test, x_test_label]
    for i, name in enumerate(for_saving):
        name = np.array(name, dtype=np.uint8)
        print(name.shape)
    '''

    #train split
    #proportion = int(len(paths) * 0.15)
    proportion = 300
    for i in range(6):
        x_train_split = x_train[i * proportion : (i+1) * proportion]
        x_train_label_split = x_train_label[i * proportion : (i+1) * proportion]
        #x_test_split = x_test[i * proportion : (i+1) * proportion]
        #x_test_label_split = x_test_label[i * proportion : (i+1) * proportion]

        x_train_split = np.array(x_train_split)
        x_train_label_split = np.array(x_train_label_split)
        #x_test_split = np.array(x_test_split)
        #x_test_label_split = np.array(x_test_label_split)

        np.save(train_save_path.format(i), x_train_split)
        np.save(train_label_save_path.format(i), x_train_label_split)
        #np.save(test_save_path.format(i), x_test_split)
        #np.save(test_label_save_path.format(i), x_test_label_split)
        print(x_train_split.shape)
        print(x_train_label_split.shape)

    x_test = np.array(x_test)
    x_test_label = np.array(x_test_label)
    np.save(test_save_path, x_test)
    np.save(test_label_save_path, x_test_label)
    print(x_test.shape)
    print(x_test_label.shape)



if __name__ == '__main__':
    main()