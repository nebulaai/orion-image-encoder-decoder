import requests
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, UpSampling2D, MaxPool2D, Activation, Dropout, BatchNormalization, ReLU
from keras import backend as K
from keras.datasets import cifar10
from keras import initializers, regularizers
from keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam
from keras.utils import multi_gpu_model
from livelossplot import PlotLossesKeras


# Create the result directory in the working directory
# Create the directory for TensorBoard variables if there is not.
if not os.path.exists("./Result"):
  os.makedirs("./Result")

# prepare the data

# Load the MNIST data
from keras.datasets import mnist

(x_train_cf10, _), (x_test_cf10, y_test_cf10) = cifar10.load_data()

print('Point 1 - x_train_cf10.shape = ', x_train_cf10.shape, ', x_test.shape = ', x_test_cf10.shape)

# Normalize all value between 0 and 1.
x_train_cf10 = x_train_cf10.astype('float32') / 255.
# the dot here is important, so that the result will be float! rather than int
x_test_cf10 = x_test_cf10.astype('float32') / 255.

print('Point 2 - x_train_cf10.shape = ', x_train_cf10.shape, ', x_test_cf10.shape = ', x_test_cf10.shape)


# Complete the baseline model of CNN-Autoencoder on CIFAR-10
# This part on the architecure of the baseline CNN autoencoder model.

# So right now, each instance is in the shape of (28x28x1)
# With the size and shape information from previous box, let's construct the baseline CNN autoencoder model.
def get_cnn_ae_baseline_cf10(encoding_dim):
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPool2D((2, 2), padding='same')(x)
    #     print('x.shape=',encoded.shape)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # autoencoder
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    print('autoencoder.summary()', autoencoder.summary())

    # encoder
    encoder = Model(input_img, encoded)

    # decoder
    encoded_input = Input(shape=encoding_dim)
    print('encoded_input.shape', encoded_input.shape)

    # I believed that there should exit a way to take out the last 8 layers from autoencoder in 1 or 2 lines.
    # To stack up layers is all the best I can figure out so far. No Aesthetics definetly, but it's functional.
    d1 = autoencoder.layers[-8]
    d2 = autoencoder.layers[-7]
    d3 = autoencoder.layers[-6]
    d4 = autoencoder.layers[-5]
    d5 = autoencoder.layers[-4]
    d6 = autoencoder.layers[-3]
    d7 = autoencoder.layers[-2]
    d8 = autoencoder.layers[-1]
    decoder = Model(encoded_input, d8(d7(d6(d5(d4(d3(d2(d1(encoded_input)))))))))
    #     print(decoder.summary())

    return encoder, decoder, autoencoder


"""Train three CNN autoencoder on MNIST CF10"""

from livelossplot import PlotLossesKeras

encoder1_cf10, decoder1_cf10, autoencoder1_cf10 = get_cnn_ae_baseline_cf10((4, 4, 8))
autoencoder1_cf10.fit(x_train_cf10, x_train_cf10,
                      epochs=3,
                      batch_size=128,
                      shuffle=True,
                      validation_data=(x_test_cf10, x_test_cf10)) #,callbacks=[PlotLossesKeras()])


# Play and tune the model of CNN-Autoencoder on CIFAR-10

def get_cnn_ae_tuned(encoding_dim):
    input_img = Input(shape=(32, 32, 3))

    x = Conv2D(90, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(60, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.000001))(x)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(40, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.000001))(x)
    x = MaxPool2D((2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    encoded = Conv2D(18, (3, 3), activation='relu', padding='same')(x)
    print('x.shape=', encoded.shape)

    x = Conv2D(18, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = BatchNormalization()(x)
    x = Conv2D(40, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(60, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(90, (3, 3), activation='relu', padding='same')(x)
    #     x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # autoencoder
    autoencoder = Model(input_img, decoded)
    # autoencoder = multi_gpu_model(autoencoder, gpus=2)   # uncoment this line if the machine has equal or more than 2 gpus
    sgd = SGD(lr=0.35, decay=1e-6, momentum=0.9, nesterov=True)
    autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')
    print(autoencoder.summary())

    # encoder
    encoder = Model(input_img, encoded)

    return encoder, autoencoder


"""Print the sample test images, their latent representation, and the decoded image"""

encoder2_cf10_tuned, autoencoder2_cf10_tuned = get_cnn_ae_tuned((4, 4, 18))
autoencoder2_cf10_tuned.fit(x_train_cf10, x_train_cf10,
                            epochs=25,
                            batch_size=100,
                            shuffle=True,
                            validation_data=(x_test_cf10, x_test_cf10))
                            # callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
                            # callbacks=[PlotLossesKeras()])

# Comparing with the baseline model in the previous box, the tuned model has been went through several trail and error modification. Some of the key modification are listed as follow:

# Plot and analysis the latent representation of CIFAR-10</font>

decoded_imgs_cf10 = autoencoder2_cf10_tuned.predict(x_test_cf10)
latent_repre_cf10 = encoder2_cf10_tuned.predict(x_test_cf10)

print(latent_repre_cf10.shape, decoded_imgs_cf10.shape)


def plot_sample(imgs, encoded_imgs, encoded_imgs_size, decoded_imgs):
    ids = np.random.randint(10000, size=10)
    plt.figure(figsize=(10, 5))
    for id in ids:
        plt.subplot(1, 3, 1)
        #     plt.gray()
        #     plt.imshow(imgs[id].reshape([32,32,3]))
        plt.imshow(imgs[id])
        plt.subplot(1, 3, 2)
        #     plt.gray()
        plt.imshow(encoded_imgs[id].reshape([-1, encoded_imgs_size]))
        plt.subplot(1, 3, 3)
        #     plt.gray()
        #     plt.imshow(decoded_imgs[id].reshape([32,32,3]))
        plt.imshow(decoded_imgs[id].reshape([32, 32, 3]))
    plt.savefig('Result/plot_sample.png')


plot_sample(x_test_cf10, latent_repre_cf10, 16, decoded_imgs_cf10)

# Further Analysis the latent representation of CIFAR-10</font>

print(latent_repre_cf10.shape)
latent_repre_faltten_cf10 = latent_repre_cf10.reshape((len(latent_repre_cf10), np.prod(latent_repre_cf10.shape[1:])))
print(latent_repre_faltten_cf10.shape)

# take summation of the latent value from the same digits in test dataset.
summation_cf10 = np.zeros((10, 288), dtype=float)
digit_count_cf10 = np.zeros(10, dtype=float)

for i in range(len(y_test_cf10)):
    current_digit = y_test_cf10[i]
    summation_cf10[current_digit, :] += latent_repre_faltten_cf10[i]
    digit_count_cf10[current_digit] += 1

# print(digit_count_cf10)

average_each_cf10 = np.zeros((10, 288), dtype=float)
for j in range(len(digit_count_cf10)):
    average_each_cf10[j, :] = summation_cf10[j, :] / digit_count_cf10[j]

namelist_CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def plot_pattern(inputdata):
    plt.figure(figsize=(10, 15))
    for digit in range(inputdata.shape[0]):
        # plt.figure(figsize=(150, 1.5))

        plt.subplot(10, 1, digit + 1)
        plt.plot(inputdata[digit, :])
        plt.title(namelist_CIFAR10[digit], color='red', fontsize=20)
        plt.ylim([np.amin(inputdata), np.amax(inputdata)])
        plt.xlim([0, 128])
        plt.axhline(0, color='black', linewidth=0.3)
        plt.axvline(21, color='black', linewidth=0.3)
        plt.axvline(42, color='black', linewidth=0.3)
        plt.axvline(63, color='black', linewidth=0.3)
        plt.axvline(84, color='black', linewidth=0.3)
        plt.axvline(105, color='black', linewidth=0.3)
    plt.savefig('Result/plot_pattern.png')


plot_pattern(average_each_cf10)

# The patterns from different objects are clearly different.
# The following part will display the plot that subtract the mean of all the value
# (filter the common pattern, keep the specific pattern)

all_average_cf10 = np.mean(average_each_cf10, axis=0)
# print(all_average[0:20])

pattern_mean_removed_cf10 = np.zeros((10, 288), dtype=float)

for j in range(len(digit_count_cf10)):
    pattern_mean_removed_cf10[j, :] = average_each_cf10[j, :] - all_average_cf10

plot_pattern(pattern_mean_removed_cf10)


# Play with the CNN layers architecture, activation functions and note your observations on the loss and the decoded image quality.

# Let's take out the airplane and bird for comparison. And discuss in detail about them.


def plot_pattern_comparison(inputdata):
    x = np.arange(inputdata.shape[1])
    #     for digit in range(inputdata.shape[0]):
    plt.figure(figsize=(20, 2))
    #         plt.subplot(1, 10, digit+1)
    #     plt.plot(x,inputdata[0,:],'r--',x,inputdata[2,:],'g*')
    plt.plot(x, inputdata[0, :], 'r--', label="Airplane")
    plt.plot(x, inputdata[2, :], 'g*', label="Bird")
    plt.legend(bbox_to_anchor=(0.05, 1), loc=2, borderaxespad=0.)

    plt.title('Airplane VS Bird', color='red', fontsize=20)
    plt.ylim([np.amin(inputdata), np.amax(inputdata)])
    plt.xlim([0, 128])
    plt.axhline(0, color='black', linewidth=0.3)
    plt.axvline(21, color='black', linewidth=0.3)
    plt.axvline(42, color='black', linewidth=0.3)
    plt.axvline(63, color='black', linewidth=0.3)
    plt.axvline(84, color='black', linewidth=0.3)
    plt.axvline(105, color='black', linewidth=0.3)
    plt.savefig('Result/plot_pattern_comparison.png')

plot_pattern_comparison(pattern_mean_removed_cf10)

# Now the latent representation with the mean removed has been ploted in one dimension (flatten) manner.
# In this way we can directly observe the difference on how the model represent the bird and airplane. First, the bird has less amplitude in comparision with the airplane. My theory to explain this is that the airplane has more deformation in the CIFAR 10 comparing to the birds. The birds in CIFAR 10 is more similier
# to each other. The second observed pattern is that: the airplane has "longer period" or we say summits are further away from
# each other in the airplane representation.


# This part we try the denoising autoencoder on the CIFAR-10 datasets

# The following section work on denoising the autoencoders
noise_factor = 0.1
x_train_noisy = x_train_cf10 + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_cf10.shape)
x_test_noisy = x_test_cf10 + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_cf10.shape)

# print(x_train_noisy[0:2,:])

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# print(x_train_noisy[0:2,:])

print(x_train_noisy.shape, x_test_noisy.shape)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_cf10[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, n + i + 1)
    plt.imshow(x_test_noisy[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('Result/figure1.png')
# plt.show()

autoencoder2_cf10_tuned.fit(x_train_noisy, x_train_cf10,
                            epochs=2,
                            batch_size=200,
                            shuffle=True,
                            validation_data=(x_test_noisy, x_test_cf10))
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
#                 callbacks = [PlotLossesKeras()])

decoded_imgs2_cf10_dnsy = autoencoder2_cf10_tuned.predict(x_test_noisy)

n = 8
plt.figure(figsize=(20, 9))
for i in range(n):
    # plot the original image
    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_test_cf10[i].reshape(32, 32, 3))
    if i == 3:
        plt.title("Original", color='red', fontsize=12)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot the decoded images (no noise added to the input).
    #     ax = plt.subplot(4, n, n+i+1)
    #     plt.imshow(decoded_imgs2_cf10[i].reshape(32, 32, 3))
    #     if i == 3:
    #         plt.title("decoded (no noise in the trainning input)",color='red',fontsize = 12)
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    # plot the images which are the original images with noise add.
    ax = plt.subplot(4, n, 2 * n + i + 1)
    plt.imshow(x_test_noisy[i].reshape(32, 32, 3))
    if i == 3:
        plt.title("original + noise in the training input", color='red', fontsize=12)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot the decoded images (with noise in the input)
    ax = plt.subplot(4, n, 3 * n + i + 1)
    plt.imshow(decoded_imgs2_cf10_dnsy[i].reshape(32, 32, 3))
    if i == 3:
        plt.title("decoded (with noise in the input)", color='red', fontsize=12)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('Result/figure2.png')
# plt.show()

# Base on the image similarity between the second row and the forth row, it is concluded that they are overall identical to each other. This mean that the autoencoder is powerful in denoising.
# The noise shown in the images as ramdon color pixel with decrease the clearity of the image. But since it is randonly distributed, it does not break the edge of object in the image. I will said that if the added noist is a continues lines in the image, it may take the model more effort to denoise that kind of noise.
# In conclude, the autoencoder show good performance on the task of filtering away the noise and keep the feature representation of the original image.
