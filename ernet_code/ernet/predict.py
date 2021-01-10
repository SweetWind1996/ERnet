import linecache
import numpy as np
import cv2
from ernet import ernet
from time import *

num_test = 100
model = ernet()
model.summary()
model.load_weights('ernet.h5')

total_time = 0
for k in range(1, num_test):
    theline = linecache.getline('test.txt', k)
    name = theline.split()
    img = cv2.imread(
        '' + '/'+name[0], 0)
    h = img.shape[0]
    w = img.shape[1]

    input_data = np.empty((7, h, w))

    for i in range(len(name)-1):
        train_img = cv2.imread(
            '' + '/'+name[i], 0)
        train_img = train_img[:h, :w]
        arr = np.asarray(train_img, dtype='float32')

        input_data[i, ...] = arr

    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=5)
    input_data = input_data/255.
    # begin_time = time()
    predict = model.predict(input_data)
    # end_time = time()
    # run_time = end_time-begin_time
    # total_time = total_time + run_time
    imagepath = "" + str(k) + ".png"
    cv2.imwrite(imagepath, predict[0, :, :, :] * 255.)
