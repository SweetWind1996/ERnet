import tensorflow.keras
# from singlegenerator import DataGenerator
# from txt_saveto_list import txt_saveto_list
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Input, Model
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense, Input, Reshape, Conv2DTranspose, AveragePooling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract


def input1(inp1):

    conv2d_1 = Conv2D(64, 3, padding='same')(inp1)
    bn_1 = BatchNormalization()(conv2d_1)
    ac_1 = ReLU()(bn_1)

    pool_1 = MaxPooling2D(2)(ac_1)

    conv2d_2 = Conv2D(128, 3, padding='same')(pool_1)
    bn_2 = BatchNormalization()(conv2d_2)
    ac_2 = ReLU()(bn_2)

    pool_2 = MaxPooling2D(2)(ac_2)

    conv2d_3 = Conv2D(256, 3, padding='same')(pool_2)
    bn_3 = BatchNormalization()(conv2d_3)
    ac_3 = ReLU()(bn_3)

    pool_3 = MaxPooling2D(2)(ac_3)

    conv2d_4 = Conv2D(512, 3, padding='same')(pool_3)
    bn_4 = BatchNormalization()(conv2d_4)
    ac_4 = ReLU()(bn_4)

    pool_4 = MaxPooling2D(2)(ac_4)

    conv2d_5 = Conv2D(256, 3, padding='same')(pool_4)
    bn_5 = BatchNormalization()(conv2d_5)
    ac_5 = ReLU()(bn_5)

    conv2d_6 = Conv2D(128, 1, padding='same')(ac_5)
    bn_6 = BatchNormalization()(conv2d_6)
    ac_6 = ReLU()(bn_6)

    flat1 = Flatten()(ac_6)

    return flat1


def input2(inp2):

    conv2d_1 = Conv2D(64, 3, padding='same')(inp2)  # 32
    bn_1 = BatchNormalization()(conv2d_1)
    ac_1 = ReLU()(bn_1)

    pool_1 = MaxPooling2D(2)(ac_1)  # 16

    conv2d_2 = Conv2D(128, 3, padding='same')(pool_1)
    bn_2 = BatchNormalization()(conv2d_2)
    ac_2 = ReLU()(bn_2)

    pool_2 = MaxPooling2D(2)(ac_2)  # 8

    conv2d_3 = Conv2D(256, 3, padding='same')(pool_2)
    bn_3 = BatchNormalization()(conv2d_3)
    ac_3 = ReLU()(bn_3)

    pool_3 = MaxPooling2D(2)(ac_3)  # 4

    conv2d_4 = Conv2D(512, 3, padding='same')(pool_3)
    bn_4 = BatchNormalization()(conv2d_4)
    ac_4 = ReLU()(bn_4)

    pool_4 = MaxPooling2D(2)(ac_4)  # 2

    conv2d_5 = Conv2D(256, 3, padding='same')(pool_4)
    bn_5 = BatchNormalization()(conv2d_5)
    ac_5 = ReLU()(bn_5)

    conv2d_6 = Conv2D(128, 1, padding='same')(ac_5)
    bn_6 = BatchNormalization()(conv2d_6)
    ac_6 = ReLU()(bn_6)

    flat2 = Flatten()(ac_6)

    return flat2


dense1 = Dense(128)
dense2 = Dense(128 * 8 * 8)
reshape = Reshape((8, 8, 128))
conv1 = Conv2D(256, 3, padding='same')
conv2 = Conv2D(1, 3, activation='relu', padding='same')
relu = ReLU()
convtans1 = Conv2DTranspose(128, 3, strides=2, padding='same')
convtans2 = Conv2DTranspose(64, 3, strides=2, padding='same')


def generate(inp3):

    x = dense1(inp3)
    x = dense2(x)
    x = relu(x)
    x = reshape(x)

    x = conv1(x)
    x = relu(x)

    x = convtans1(x)
    x = relu(x)

    x = convtans2(x)
    x = relu(x)

    x = conv2(x)
    x = relu(x)

    return x


def model():
    inp = Input(shape=(32, 32, 1))
    fe1 = input1(inp)
    fe2 = input2(inp)
    g1 = generate(fe1)
    g2 = generate(fe2)
    model = Model(inp, (g1, g2))
    return model


model = model()
model.summary()
model.compile(
    loss='mse',
    optimizer=optimizers.RMSprop(lr=1e-5)
)

path_train = 'train.txt'
train_txt_len = len(open(path_train, 'r').readlines())
temp_list_t = []
train_list = txt_saveto_list(path_train, train_txt_len, temp_list_t)

path_val = 'val.txt'
val_txt_len = len(open(path_val, 'r').readlines())
temp_list_v = []
val_list = txt_saveto_list(path_val, val_txt_len, temp_list_v)

params1 = {'dim1': (32, 32),
           'dim2': (32, 32),
           'batch_size': 128,
           'train_list': train_list,
           'n_channels': 1,
           'shuffle': True,
           'traindata_dir': '2D/traindata',
           'labeldata_dir': '2D/trainlabel'}

params2 = {'dim1': (32, 32),
           'dim2': (32, 32),
           'batch_size': 128,
           'train_list': val_list,
           'n_channels': 1,
           'shuffle': True,
           'traindata_dir': '2D/valdata',
           'labeldata_dir': '2D/vallabel'}
training_generator = DataGenerator(**params1)
val_generator = DataGenerator(**params2)

checkpoint = ModelCheckpoint(
    filepath='Ranknet.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only='True',
    mode='auto',
    period=1
)
checkpoint = [checkpoint]
history = model.fit_generator(
    generator=training_generator, validation_data=val_generator, epochs=500, callbacks=checkpoint)
