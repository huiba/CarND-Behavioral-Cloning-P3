import csv
from re import T
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers 
import sklearn
from sklearn.model_selection import train_test_split
import json
import random

center_im_list = []
left_im_list = []
right_im_list = []
label_list = []

all_images = []
all_labels = []
#path_prefix = ['./data/new_data', './data/new_data_recovery', './data/new_data_smooth', './data/new_data_curve']
#path_prefix = ['./data/new_data', './data/new_data_recovery', './data/new_data_smooth']
path_prefix = ['./data/data/']
correction = 0.2
# get data from all the three cameras
for pp in path_prefix:
    lines = []
    with open(os.path.join(pp, 'driving_log.csv')) as f:
        reader = csv.reader(f)
        for l in reader:
            lines.append(l)
    for line in lines[1:]:
        cp = os.path.join(pp, line[0].strip())
        lp = os.path.join(pp, line[1].strip())
        rp = os.path.join(pp, line[2].strip())
        l = float(line[3])
        all_images.append(cp)
        all_labels.append(l)
        all_images.append(lp)
        all_labels.append(l + correction)
        all_images.append(rp)
        all_labels.append(l - correction)
        #center_im_list.append(cp)
        #left_im_list.append(lp)
        #right_im_list.append(rp)
        #label_list.append(float(line[3]))

def data_generator(samples, batch_size=32):
    """
    assume samples: (img paths, labels) 
    """
    X, y = samples 
    num_samples = len(X)
    while True:
        X, y = sklearn.utils.shuffle(X, y)
        for offset in range(0, num_samples, batch_size):
            X_path_batch = X[offset:offset+batch_size]
            X_batch = []
            y_batch_bef = y[offset:offset+batch_size]
            y_batch = []
            for b_im, b_label in zip(X_path_batch, y_batch_bef):
                image = cv2.imread(b_im) 
                # random flipping
                if random.choice((True, False)):
                    image = image[:,::-1,:]
                    b_label = - b_label
                X_batch.append(image)
                y_batch.append(b_label)
            X_batch = np.array(X_batch)
            X_batch = np.array(X_batch, dtype=np.float32)
            y_batch = np.array(y_batch)
            yield tuple(sklearn.utils.shuffle(X_batch, y_batch))

#X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(center_im_list, label_list, test_size=0.10)
X_train_list, X_test_list, y_train_list, y_test_list = train_test_split(all_images, all_labels, test_size=0.10)

batch_size = 32
test_batch_size = 16
train_gen = data_generator((X_train_list, y_train_list), batch_size=batch_size)
test_gen = data_generator((X_test_list, y_test_list), batch_size=test_batch_size)

num_train_data = len(X_train_list) - 1 
steps_per_epoch_train =  num_train_data / batch_size
num_test_data = len(X_test_list) - 1 
steps_per_epoch_test =  num_test_data / test_batch_size

#train_ds = tf.data.Dataset.from_generator(lambda:train_gen, 
#                                          output_signature=(tf.TensorSpec(shape=(None, 160, 320, 3), dtype=tf.float32),
#                                                            tf.TensorSpec(shape=(None), dtype=tf.float32)))
                                         
#test_ds = tf.data.Dataset.from_generator(lambda:test_gen,
#                                          output_signature=(tf.TensorSpec(shape=(None, 160, 320, 3), dtype=tf.float32),
#                                                            tf.TensorSpec(shape=(None), dtype=tf.float32)))


#def get_dataset(X, y, train=True, batch_size=32, shuffle_size=1000):
#    AUTOTUNE = tf.data.AUTOTUNE
#    ds = tf.data.Dataset.from_tensor_slices((X, y))
#    ds = ds.map(lambda x, y: (tf.io.read_file(x), y))
#    ds = ds.map(lambda x, y: (tf.io.decode_jpeg(x), y))
#    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
#    #ds = ds.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y),
    #            num_parallel_calls=AUTOTUNE)
    #ds = ds.map(lambda x, y: (tf.image.per_image_standardization(x), y),
    #            num_parallel_calls=AUTOTUNE)
#    if train:
        #ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y),
        #                        num_parallel_calls=AUTOTUNE)
#        ds = ds.shuffle(shuffle_size)
#    ds = ds.batch(batch_size) 
    #return ds.prefetch(buffer_size=AUTOTUNE)
#    return ds


def get_model():
    model = Sequential()
    model.add(layers.Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
    model.add(layers.Lambda(lambda x: x / 255.0 - 0.5))
    model.add(layers.Conv2D(8, 3, activation='relu'))
    model.add(layers.Conv2D(16, 3, activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.Conv2D(32, 3, activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Flatten())
    #model.add(layers.Dense(512, activation='relu', kernel_regularizer='l2'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    return model

def get_model2(l2=0.001):
    model = Sequential()
    model.add(layers.Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
    model.add(layers.Lambda(lambda x: x / 255.0 - 0.5))
    # 160 x 320 x 3
    model.add(layers.Conv2D(24, 5, strides=2, padding='valid', activation='relu'))
    # 78 x 158 x 24
    model.add(layers.Conv2D(36, 5, strides=2, padding='valid', activation='relu'))
    # 37 x 77 x 36 
    model.add(layers.Conv2D(48, 5, strides=2, padding='valid', activation='relu'))
    # 17 x 37 x 48
    model.add(layers.Conv2D(64, 3, strides=1, padding='valid', activation='relu'))
    # 8 x 18 x 64
    model.add(layers.Conv2D(64, 3, strides=1, padding='valid', activation='relu'))
    #model.add(layers.Dropout(0.3))
    # 3 x 8 x 64
    model.add(layers.Flatten())
    #model.add(layers.Dense(512, activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(12, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    return model
    
if __name__ == '__main__':
    #train_ds = get_dataset(X_train_list, y_train_list)
    #from IPython impo
    #test_ds = get_dataset(X_test_list, y_test_list, train=False)
    #model = get_model()
    model = get_model2()
    model.summary()
    epochs = 10
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.001))
    #model.load_weights('weights/model1_gen_flip_clr_crop_udacity_data\weights.epoch10.h5')
    name = 'model2_gen_flip_clr_crop_udacity_data'
    #tf.keras.utils.plot_model(model, to_file='model1.png', show_layer_names=False)
    weights_path = os.path.join('weights', name, 'weights.epoch{epoch:02d}.h5')
    callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=weights_path, save_freq='epoch')
            #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    ]
    with open(f'weights/{name}.json', 'w') as f:
        f.write(json.dumps(json.loads(model.to_json()), indent=4))
    history = model.fit(x=train_gen, 
                     validation_data=test_gen, 
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_steps=steps_per_epoch_test,
                        steps_per_epoch=steps_per_epoch_train)
    #save_path = f'./data/{name}_e{epochs}.h5'
    #model.save(save_path)
    #print(f'Model saved: {save_path}')
