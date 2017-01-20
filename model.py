import scipy.misc
import random
import csv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Input, Activation, ELU
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

xs = []
ys = []

with open("driving_log.csv", 'r') as f:
    image_reader = csv.reader(f, delimiter=',')
    next(image_reader, None)

    # Read from csv for center image name and steering angles
    for line in image_reader:
        xs.append(line[0])
        ys.append(float(line[3]))

# get number of images
num_images = len(xs)

# shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

image_size = 64

dataset = np.ndarray(shape=(num_images, image_size, image_size, 3),
                     dtype=np.float32)
labelset = np.ndarray(shape=(num_images, 1), dtype=np.float32)
for i in range(0, num_images):
    # Image size is 160 x 320
    # Crop out top 40 and bottom 20 pixels of the image
    # Image became 100 x 320
    # Resize it to 64 x 64
    # Normalize the image then
    image = scipy.misc.imresize(scipy.misc.imread(xs[i])[-120:-20],
                                [image_size, image_size]) / 255.0
    dataset[i, :, :, :] = image
    labelset[i] = ys[i]

Y_train = labelset
X_train = dataset

model = Sequential()
# Convolution layer 1, 3 x 3 filter, Maxpool 2 x 2
model.add(Conv2D(32, 3, 3, input_shape=(image_size, image_size, 3), name='conv1'))
model.add(MaxPooling2D((2, 2)))
model.add(ELU())

# Convolution layer 2, 3 x 3 filter, Maxpool 2 x 2
model.add(Conv2D(64, 3, 3, name='conv2'))
model.add(MaxPooling2D((2, 2)))
model.add(ELU())

# Convolution layer 3, 3 x 3 filter, Maxpool 2 x 2
model.add(Conv2D(128, 3, 3, name='conv3'))
model.add(MaxPooling2D((2, 2)))
model.add(ELU())

# Convolution layer 3, 3 x 3 filter, Maxpool 2 x 2
model.add(Conv2D(256, 3, 3, name='conv4'))
model.add(MaxPooling2D((2, 2)))
model.add(ELU())

model.add(Flatten())

# Fully connected layer 1
model.add(Dense(512, name='FC1', init='he_normal'))
model.add(ELU())

# Fully connected layer 2
model.add(Dense(128, name='FC2', init='he_normal'))
model.add(ELU())

# Fully connected layer 3
model.add(Dense(16, name='FC3', init='he_normal'))
model.add(ELU())

model.add(Dense(1, name='output'))

model.summary()

# Adam optimizer
# Set learning rate (lr) to 0.0001
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(adam, "mse")

# Setup tensorboard
tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)

# Train model
# shuffle turned on
# validation_split set to 25%
history = model.fit(X_train, Y_train,
                    batch_size=32, nb_epoch=4,
                    verbose=1, validation_split=0.25, shuffle=True, callbacks=[tb])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
