import os
import tensorflow.keras
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class DataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, dim1, dim2, batch_size, train_list, n_channels, shuffle, traindata_dir, labeldata_dir):

        self.dim1 = dim1
        self.dim2 = dim2
        self.batch_size = batch_size
        self.train_list = train_list
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.traindata_dir = traindata_dir
        self.labeldata_dir = labeldata_dir
        self.on_epoch_end()

    def on_epoch_end(self):

        if self.shuffle == True:
            np.random.shuffle(self.train_list)

    def data_per_batch(self):

        for i in range(self.batch_size):
            a = random.randrange(len(self.train_list))
            data = self.train_list[a]
            yield data

    def __data_generation(self, train_list_temp):

        index = 0

        x = np.empty((self.batch_size, *self.dim1, self.n_channels))
        y = np.empty((self.batch_size, *self.dim2, self.n_channels))
        for item in train_list_temp:
            img_in_row = Image.open(self.traindata_dir + '/' + item[0])
            img_to_arr = np.asarray(img_in_row)
            img_to_arr = np.expand_dims(img_to_arr, axis=2)
            x[index, ...] = img_to_arr
            label_per_row = Image.open(self.labeldata_dir + '/' + item[0])
            label_to_arr = np.asarray(label_per_row)
            label_to_arr = np.expand_dims(label_to_arr, axis=2)
            y[index, ...] = label_to_arr
            index = index + 1
        return x/255., [x/255., y/255.]

    def __getitem__(self, index):

        train_list__temp = self.data_per_batch()
        train_data, label_data = self.__data_generation(
            train_list__temp)
        return train_data, label_data

    def __len__(self):

        return int(np.floor(len(self.train_list) / self.batch_size))
