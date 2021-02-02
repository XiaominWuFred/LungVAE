
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../pairImg")
from loadData import loadData

from pairImg import pairImg

def load_data():
    #custom Mask
    loadImg = loadData('../../pairImg/images', '../../pairImg/newMasks', '../../pairImg/train.csv', 0.8)
    #binary mask
    #loadImg = loadData('../../pairImg/images', '../../pairImg/masks', '../../pairImg/train.csv', 0.8)

    x_trainN = loadImg.trainN
    x_testN = loadImg.testN
    x_trainP = loadImg.trainP
    x_testP = loadImg.testP
    '''
    s = x_test[:10]
    for i in range(len(s)):
        img2avg = s[i]
        plt.imshow(img2avg)
        plt.savefig('vae_mlp/img'+str(i)+'.png')
    '''
    y_trainN = np.ones(800)
    y_testN = np.ones(200)
    y_trainP = np.ones(800)
    y_testP = np.ones(200)

    return (x_trainN, y_trainN), (x_testN, y_testN),(x_trainP, y_trainP), (x_testP, y_testP)


# Model configuration
img_width, img_height = 128, 128
batch_size = 64
no_epochs = 20
validation_split = 0.2
verbosity = 1
max_norm_value = 2.0
noise_factor = 0.55
number_of_visualizations = 20

custom = True

# Load MNIST dataset
if custom == True:
    # load custom dataset
    (x_trainN, y_trainN), (x_testN, y_testN),(x_trainP, y_trainP), (x_testP, y_testP) = load_data()
    x_trainN = x_trainN.reshape((x_trainN.shape[0],x_trainN.shape[1],x_trainN.shape[2],1))
    x_testN = x_testN.reshape((x_testN.shape[0], x_testN.shape[1], x_testN.shape[2], 1))
    x_testP = x_testP.reshape((x_testP.shape[0], x_testP.shape[1], x_testP.shape[2], 1))
    x_trainP = x_trainP.reshape((x_trainP.shape[0], x_trainP.shape[1], x_trainP.shape[2], 1))

else:
    (input_train, target_train), (input_test, target_test) = mnist.load_data()



# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if custom:
    noisy_input = x_trainN
    noisy_input_test = x_testP
    pure = x_trainP
    pure_test = x_testP

    target_test = y_testP

    input_shape = (img_width, img_height, 1)

else:
    if K.image_data_format() == 'channels_first':
        input_train = input_train.reshape(input_train.shape[0], 1, img_width, img_height)
        input_test = input_test.reshape(input_test.shape[0], 1, img_width, img_height)
        input_shape = (1, img_width, img_height)
    else:
        input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
        input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
        input_shape = (img_width, img_height, 1)

        # Parse numbers as floats
        input_train = input_train.astype('float32')
        input_test = input_test.astype('float32')

        # Normalize data
        input_train = input_train / 255
        input_test = input_test / 255
        # Add noise
        pure = input_train
        pure_test = input_test
        noise = np.random.normal(0, 1, pure.shape)
        noise_test = np.random.normal(0, 1, pure_test.shape)
        noisy_input = pure + noise_factor * noise
        noisy_input_test = pure_test + noise_factor * noise_test





# Create the model
model = Sequential()
model.add(Conv2D(128, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(32, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(32, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(16, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(16, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(8, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(8, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(16, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(16, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(32, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(32, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(64, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2DTranspose(128, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(1, kernel_size=(5,5), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))

model.summary()

# Compile and fit data
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(noisy_input, pure,
                epochs=no_epochs,
                batch_size=batch_size,
                validation_split=validation_split)

# Generate denoised images
samples = noisy_input_test[:number_of_visualizations]
targets = target_test[:number_of_visualizations]
denoised_images = model.predict(samples)
#denoised_images = denoised_images - samples
#denoised_images[denoised_images<0] = 0


# Plot denoised images
for i in range(0, number_of_visualizations):
  # Get the sample and the reconstruction
  noisy_image = noisy_input_test[i][:, :, 0]
  pure_image  = pure_test[i][:, :, 0]
  denoised_image = denoised_images[i][:, :, 0]
  input_class = targets[i]
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 3)
  fig.set_size_inches(8, 3.5)
  # Plot sample and reconstruciton
  axes[0].imshow(noisy_image)
  axes[0].set_title('X-ray image')
  axes[1].imshow(pure_image)
  axes[1].set_title('original mask image')
  axes[2].imshow(denoised_image)
  axes[2].set_title('regenerated mask image')
  #fig.suptitle(f'MNIST target = {input_class}')
  plt.savefig('../../dvaeOut/sampleVisCustomMask'+str(i)+'.png')
  plt.show()

