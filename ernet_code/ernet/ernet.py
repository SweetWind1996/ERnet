import tensorflow.keras
import tensorflow as tf
from DataGenerator import DataGenerator
from ValidationGenerator import ValGenerator
from txt_saveto_list import txt_saveto_list
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Add, Conv3D, Conv3DTranspose, ReLU, BatchNormalization, add, Lambda, Maximum, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, Maximum, MaxPooling2D, ReLU, BatchNormalization, add, Lambda, Maximum, Conv2D, Conv2DTranspose, Average
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling3D, GlobalMaxPooling3D, Dense, Multiply, Concatenate


def res(x, filters, ksize1, ksize2):
    x = layers.Conv3D(filters, ksize1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    res = layers.Conv3D(filters, ksize2, padding='same')(x)
    x = Add()([x, res])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def upres(x, filters, upsize, strides, ksize1, ksize2, cat):
    x = layers.Conv3DTranspose(filters, upsize, strides, padding='same')(x)
    x = layers.Concatenate()([cat, x])
    x = layers.Conv3D(filters, ksize1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    res = layers.Conv3D(filters, ksize2, padding='same')(x)
    x = Add()([x, res])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def res2d(x, filters, ksize1, ksize2, strides):
    x = Conv2D(filters, ksize1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    res = Conv2D(filters, ksize1, padding='same')(x)
    x = Add()([x, res])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, ksize2, strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def upres2d(x, filters, ksize1, strides, cat, ksize2):
    x = Conv2DTranspose(filters, ksize1, strides, padding='same')(x)
    x = layers.Concatenate()([cat, x])
    x = layers.Conv2D(filters, ksize2, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    res = Conv2D(filters, ksize2, padding='same')(x)
    x = Add()([res, x])
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def ernet():
    inputs = Input(shape=(7, None, None, 1))  # 32

    fea3d = res(inputs, 64, (1, 3, 3), (3, 1, 1))
    att_avg3d = GlobalAveragePooling3D()(fea3d)
    att_max3d = GlobalMaxPooling3D()(fea3d)
    att_avg3d = Dense(64, activation='relu')(att_avg3d)
    att_avg3d = Dense(64, activation='relu')(att_avg3d)
    att_max3d = Dense(64, activation='relu')(att_max3d)
    att_max3d = Dense(64, activation='relu')(att_max3d)
    att3d = Add()([att_avg3d, att_max3d])
    fea3d = Multiply()([fea3d, att3d])
    att = layers.Conv3D(128, (7, 1, 1),  activation='relu')(fea3d)
    fea = Lambda(lambda x: x[:, 0, :, :, :])(att)
    att_avg = GlobalAveragePooling2D()(fea)
    att_max = GlobalMaxPooling2D()(fea)
    att_avg = Dense(128, activation='relu')(att_avg)
    att_avg = Dense(128, activation='relu')(att_avg)
    att_max = Dense(128, activation='relu')(att_max)
    att_max = Dense(128, activation='relu')(att_max)
    att = Add()([att_avg, att_max])
    fea = Multiply()([fea, att])
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(fea)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(fea)
    att = Concatenate(axis=3)([avg_pool, max_pool])
    att = Conv2D(filters=1, kernel_size=3, padding='same',
                 activation='sigmoid')(att)
    fea = Multiply()([fea, att])

    res1 = res(inputs, 64, (1, 3, 3), (3, 1, 1))
    Pooling1 = layers.MaxPooling3D((1, 2, 2))(res1)  # 16
    res2 = res(Pooling1, 128, (1, 3, 3), (3, 1, 1))
    Pooling2 = layers.MaxPooling3D((1, 2, 2))(res2)  # 8
    res3 = res(Pooling2, 256, (1, 3, 3), (3, 1, 1))
    Pooling3 = layers.MaxPooling3D((1, 2, 2))(res3)  # 4
    res4 = res(Pooling3, 256, (1, 3, 3), (3, 1, 1))
    Pooling4 = layers.MaxPooling3D((1, 2, 2))(res4)  # 2

    Conv3D_5 = layers.Conv3D(
        256, (1, 1, 1), padding='same', activation='relu')(Pooling4)  # 2

    upres1 = upres(Conv3D_5, 256, (1, 4, 4), (1, 2, 2),
                   (1, 3, 3), (3, 1, 1), res4)
    upres2 = upres(upres1, 256, (1, 4, 4), (1, 2, 2),
                   (1, 3, 3), (3, 1, 1), res3)
    upres3 = upres(upres2, 128, (1, 4, 4), (1, 2, 2),
                   (1, 3, 3), (3, 1, 1), res2)
    upres4 = upres(upres3, 64, (1, 4, 4), (1, 2, 2),
                   (1, 3, 3), (3, 1, 1), res1)

    inter = layers.Conv3D(1, (7, 1, 1),  activation='relu')(upres4)  # 32
    inter = Add()([att, inter])

    x = Lambda(lambda x: x[:, 0, :, :, :])(inter)

    res2d0 = res2d(x, 64, 1, 1, 1)
    res2d1 = res2d(res2d0, 64, 3, 4, 2)
    res2d2 = res2d(res2d1, 128, 3, 4, 2)
    res2d3 = res2d(res2d2, 256, 3, 4, 2)
    res2d4 = res2d(res2d3, 256, 3, 4, 2)

    x = Conv2D(filters=512, kernel_size=(1, 1), strides=1,
               padding='same', activation='relu')(res2d4)  # 2

    upres2d1 = upres2d(x, 256, 4, 2, res2d3, 3)
    upres2d2 = upres2d(upres2d1, 256, 4, 2, res2d2, 3)
    upres2d3 = upres2d(upres2d2, 128, 4, 2, res2d1, 3)
    upres2d4 = upres2d(upres2d3, 64, 4, 2, res2d0, 3)

    x = Conv2D(filters=1, kernel_size=1, strides=1)(upres2d4)
    # x = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(upres2d4)  # 32

    model = Model(inputs, x)
    return model


model = ernet()

model.summary()


model.compile(
    loss="mse",
    optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6)
)


path_train = '/home/gzs/Desktop/zpj/space/train.txt'
path_val = '/home/gzs/Desktop/zpj/space/val.txt'
train_txt_len = len(open(path_train, 'r').readlines())
val_txt_len = len(open(path_val, 'r').readlines())
temp_list_t = []
temp_list_v = []
train_list = txt_saveto_list(path_train, train_txt_len, temp_list_t)
val_list = txt_saveto_list(path_val, val_txt_len, temp_list_v)

trainparams = {'dim1': (7, 32, 32),
               'dim2': (32, 32),
               'batch_size': 64,
               'train_list': train_list,
               'n_channels': 1,

               'shuffle': True,
               'traindata_dir': '/home/gzs/Desktop/zpj/space/traindata',
               'labeldata_dir': '/home/gzs/Desktop/zpj/space/trainlabel'}

valparams = {'dim1': (7, 32, 32),
             'dim2': (32, 32),
             'batch_size': 64,
             'train_list': val_list,
             'n_channels': 1,
             'shuffle': True,
             'traindata_dir': '/home/gzs/Desktop/zpj/space/valdata',
             'labeldata_dir': '/home/gzs/Desktop/zpj/space/vallabel'}

training_generator = DataGenerator(**trainparams)
validation_generator = DataGenerator(**valparams)

checkpoint = ModelCheckpoint(
    filepath='ernet.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only='True',
    mode='auto',
    period=1
)
checkpoint = [checkpoint]
history = model.fit_generator(
    generator=training_generator, validation_data=validation_generator, epochs=20, callbacks=checkpoint)

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'r', label='training loss')
# plt.plot(epochs, val_loss, 'b', label='validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
