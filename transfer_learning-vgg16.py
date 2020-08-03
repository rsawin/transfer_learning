# Source of code: "A Comprehensive Hands-on Guide to Transfer Learning with Real-World Applications in Deep Learning"
# https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a
# Github:  https://github.com/dipanjanS/hands-on-transfer-learning-with-python/tree/master/notebooks

# Building Datasets

# To start, download the train.zip file from the dataset page and store it in your local system. Once downloaded,
# unzip it into a folder. This folder will contain 25,000 images of dogs and cats; that is, 12,500 images per
# category. While we can use all 25,000 images and build some nice models on them, if you remember, our problem
# objective includes the added constraint of having a small number of images per category. Let’s build our own
# dataset for this purpose.

import glob
import os
import shutil

import numpy as np

np.random.seed(42)

files = glob.glob('train/*')

cat_files = [fn for fn in files if 'cat' in fn]
dog_files = [fn for fn in files if 'dog' in fn]
len(cat_files), len(dog_files)

# We can verify with the preceding output that we have 12,500 images for each category. Let’s now build our smaller
# dataset, so that we have 3,000 images for training, 1,000 images for validation, and 1,000 images for our test
# dataset (with equal representation for the two animal categories).

cat_train = np.random.choice(cat_files, size=1500, replace=False)
dog_train = np.random.choice(dog_files, size=1500, replace=False)
cat_files = list(set(cat_files) - set(cat_train))
dog_files = list(set(dog_files) - set(dog_train))

cat_val = np.random.choice(cat_files, size=500, replace=False)
dog_val = np.random.choice(dog_files, size=500, replace=False)
cat_files = list(set(cat_files) - set(cat_val))
dog_files = list(set(dog_files) - set(dog_val))

cat_test = np.random.choice(cat_files, size=500, replace=False)
dog_test = np.random.choice(dog_files, size=500, replace=False)

print('Cat datasets:', cat_train.shape, cat_val.shape, cat_test.shape)
print('Dog datasets:', dog_train.shape, dog_val.shape, dog_test.shape)

# Now that our datasets have been created, let’s write them out to our disk in separate folders, so that we can
# come back to them anytime in the future without worrying if they are present in our main memory.

train_dir = 'training_data'
val_dir = 'validation_data'
test_dir = 'test_data'

train_files = np.concatenate([cat_train, dog_train])
validate_files = np.concatenate([cat_val, dog_val])
test_files = np.concatenate([cat_test, dog_test])

os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None

for fn in train_files:
    shutil.copy(fn, train_dir)

for fn in validate_files:
    shutil.copy(fn, val_dir)

for fn in test_files:
    shutil.copy(fn, test_dir)

# Preparing Datasets
# Before we jump into modeling, let’s load and prepare our datasets. To start with, we load up some basic dependencies.

import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# %matplotlib inline

# Let’s now load our datasets, using the following code snippet.
IMG_DIM = (150, 150)

train_files = glob.glob('training_data/*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in train_files]

validation_files = glob.glob('validation_data/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in validation_files]

print('Train dataset shape:', train_imgs.shape,
      '\tValidation dataset shape:', validation_imgs.shape)

# We can clearly see that we have 3000 training images and 1000 validation images. Each image is of size 150 x 150
# and has three channels for red, green, and blue (RGB), hence giving each image the (150, 150, 3) dimensions.
# We will now scale each image with pixel values between (0, 255) to values between (0, 1) because deep learning
# models work really well with small input values.

train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

print(train_imgs[0].shape)
array_to_img(train_imgs[0])

# The preceding output shows one of the sample images from our training dataset. Let’s now set up some basic
# configuration parameters and also encode our text class labels into numeric values (otherwise, Keras will
# throw an error).

batch_size = 30
num_classes = 2
epochs = 30
input_shape = (150, 150, 3)

# encode text category labels
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)

print(train_labels[1495:1505], train_labels_enc[1495:1505])


# Simple CNN Model from Scratch

# We will start by building a basic CNN model with three convolutional layers, coupled with max pooling for
# auto-extraction of features from our images and also downsampling the output convolution feature maps.

# We assume you have enough knowledge about CNNs and hence, won’t cover theoretical details. Feel free to refer to
# my book or any other resources on the web which explain convolutional neural networks! Let’s leverage Keras and
# build our CNN model architecture now.

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])

model.summary()

# We use a batch_size of 30 and our training data has a total of 3,000 samples, which indicates
# that there will be a total of 100 iterations per epoch. We train the model for a total of 30 epochs
# and validate it consequently on our validation set of 1,000 images.

history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# Looks like our model is kind of overfitting, based on the training and validation accuracy values.
# We can plot our model accuracy and errors using the following snippet to get a better perspective.

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1, 31))
ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 31, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 31, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

# CNN Model with Regularization

# Let’s improve upon our base CNN model by adding in one more convolution layer, another dense hidden layer. Besides
# this, we will add dropout of 0.3 after each hidden dense layer to enable regularization. Basically, dropout is a
# powerful method of regularizing in deep neural nets. It can be applied separately to both input layers and the
# hidden layers. Dropout randomly masks the outputs of a fraction of units from a layer by setting their output to
# zero (in our case, it is 30% of the units in our dense layers).

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])

history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# You can clearly see from the preceding outputs that we still end up overfitting the model, though it takes slightly
# longer and we also get a slightly better validation accuracy of around 78%, which is decent but not amazing. The
# reason for model overfitting is because we have much less training data and the model keeps seeing the same
# instances over time across each epoch. A way to combat this would be to leverage an image augmentation strategy to
# augment our existing training data with images that are slight variations of the existing images. We will cover this
# in detail in the following section. Let’s save this model for the time being so we can use it later to evaluate its
# performance on the test data.

# Save the model
model.save('cats_dogs_basic_cnn.h5')

# CNN Model with Image Augmentation

# Let’s improve upon our regularized CNN model by adding in more data using a proper image augmentation strategy.
# Since our previous model was trained on the same small sample of data points each time, it wasn’t able to
# generalize well and ended up overfitting after a few epochs. The idea behind image augmentation is that we follow
# a set process of taking in existing images from our training dataset and applying some image transformation
# operations to them, such as rotation, shearing, translation, zooming, and so on, to produce new, altered versions
# of existing images. Due to these random transformations, we don’t get the same images each time, and we will
# leverage Python generators to feed in these new images to our model during training.

# The Keras framework has an excellent utility called ImageDataGenerator that can help us in doing all the
# preceding operations. Let’s initialize two of the data generators for our training and validation datasets.

train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1. / 255)

img_id = 2595
cat_generator = train_datagen.flow(train_imgs[img_id:img_id + 1], train_labels[img_id:img_id + 1],
                                   batch_size=1)
cat = [next(cat_generator) for i in range(0, 5)]
fig, ax = plt.subplots(1, 5, figsize=(16, 6))
print('Labels:', [item[1][0] for item in cat])
l = [ax[i].imshow(cat[i][0][0]) for i in range(0, 5)]

img_id = 1991
dog_generator = train_datagen.flow(train_imgs[img_id:img_id + 1], train_labels[img_id:img_id + 1],
                                   batch_size=1)
dog = [next(dog_generator) for i in range(0, 5)]
fig, ax = plt.subplots(1, 5, figsize=(15, 6))
print('Labels:', [item[1][0] for item in dog])
l = [ax[i].imshow(dog[i][0][0]) for i in range(0, 5)]

train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)
input_shape = (150, 150, 3)


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers

model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50,
                              verbose=1)

# We reduce the default learning rate by a factor of 10 here for our optimizer to prevent the model from getting stuck
# in a local minima or overfit, as we will be sending a lot of images with random transformations. To train the
# model, we need to slightly modify our approach now, since we are using data generators. We will leverage the
# fit_generator(…) function from Keras to train this model. The train_generator generates 30 images each time,
# so we will use the steps_per_epoch parameter and set it to 100 to train the model on 3,000 randomly generated
# images from the training data for each epoch. Our val_generator generates 20 images each time so we will set the
# validation_steps parameter to 50 to validate our model accuracy on all the 1,000 validation images
# (remember we are not augmenting our validation dataset).

model.save('cats_dogs_cnn_img_aug.h5')


# Thus, we are mostly concerned with leveraging the convolution blocks of the VGG-16 model and then flattening the
# final output (from the feature maps) so that we can feed it into our own dense layers for our classifier.

# Pre-trained CNN model as a Feature Extractor

# Let’s leverage Keras, load up the VGG-16 model, and freeze the convolution blocks so that
# we can use it as just an image feature extractor.
from keras.applications import vgg16
from vgg16 import VGG16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                  input_shape=input_shape)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model.trainable = False
for layer in vgg_model.layers:
    layer.trainable = False

import pandas as pd

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
df = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
df

# It is quite clear from the preceding output that all the layers of the VGG-16 model are frozen, which is good,
# because we don’t want their weights to change during model training. The last activation feature map in the
# VGG-16 model (output from block5_pool) gives us the bottleneck features, which can then be flattened and fed to a
# fully connected deep neural network classifier. The following snippet shows what the bottleneck features look like
# for a sample image from our training data.

bottleneck_feature_example = vgg.predict(train_imgs_scaled[0:1])
print(bottleneck_feature_example.shape)
plt.imshow(bottleneck_feature_example[0][:, :, 0])


# We flatten the bottleneck features in the vgg_model object to make them ready to be fed to our fully connected
# classifier. A way to save time in model training is to use this model and extract out all the features from our
# training and validation datasets and then feed them as inputs to our classifier. Let’s extract out the bottleneck
# features from our training and validation sets now.

def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features


train_features_vgg = get_bottleneck_features(vgg_model, train_imgs_scaled)
validation_features_vgg = get_bottleneck_features(vgg_model, validation_imgs_scaled)

print('Train Bottleneck Features:', train_features_vgg.shape,
      '\tValidation Bottleneck Features:', validation_features_vgg.shape)


# The preceding output tells us that we have successfully extracted the flattened bottleneck features of
# dimension 1 x 8192 for our 3,000 training images and our 1,000 validation images. Let’s build the architecture
# of our deep neural network classifier now, which will take these features as input.
from keras.layers import Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

input_shape = vgg_model.output_shape[1]

model = Sequential()
model.add(InputLayer(input_shape=(input_shape,)))
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

model.summary()

# Just like we mentioned previously, bottleneck feature vectors of size 8192 serve as input to our classification
# model. We use the same architecture as our previous models here with regard to the dense layers. Let’s train
# this model now.

history = model.fit(x=train_features_vgg, y=train_labels_enc,
                    validation_data=(validation_features_vgg, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

# We get a model with a validation accuracy of close to 88%, almost a 5–6% improvement from our basic CNN model with
# image augmentation, which is excellent. The model does seem to be overfitting though. There is a decent gap between
# the model train and validation accuracy after the fifth epoch, which kind of makes it clear that the model is
# overfitting on the training data after that. But overall, this seems to be the best model so far. Let’s try using
# our image augmentation strategy on this model. Before that, we save this model to disk using the following code.

model.save('cats_dogs_tlearn_basic_cnn.h5')


# Pre-trained CNN model as a Feature Extractor with Image Augmentation
# We will leverage the same data generators for our train and validation datasets that we used before.
# The code for building them is depicted as follows for ease of understanding.
train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)


# Let’s now build our deep learning model and train it. We won’t extract the bottleneck features like last time since
# we will be training on data generators; hence, we will be passing the vgg_model object as an input to our own model.
# We bring the learning rate slightly down since we will be training for 100 epochs and don’t want to make any sudden
# abrupt weight adjustments to our model layers. Do remember that the VGG-16 model’s layers are still frozen here,
# and we are still using it as a basic feature extractor only.

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50,
                              verbose=1)

model.save('cats_dogs_tlearn_img_aug_cnn.h5')



# We will now fine-tune the VGG-16 model to build our last classifier, where we will unfreeze blocks 4 and 5,
# as we depicted in our block diagram earlier.
#
# Pre-trained CNN model with Fine-tuning and Image Augmentation
# We will now leverage our VGG-16 model object stored in the vgg_model variable and
# unfreeze convolution blocks 4 and 5 while keeping the first three blocks frozen.
# The following code helps us achieve this.

vgg_model.trainable = True

set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])


# You can clearly see from the preceding output that the convolution and pooling layers pertaining to blocks 4 and 5
# are now trainable. This means the weights for these layers will also get updated with backpropagation in each epoch
# as we pass each batch of data. We will use the same data generators and model architecture as our previous model and
# train our model. We reduce the learning rate slightly, since we don’t want to get stuck at any local minimal, and
# we also do not want to suddenly update the weights of the trainable VGG-16 model layers by a big factor that might
# adversely affect the model.


train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow(train_imgs, train_labels_enc, batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=20)

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import optimizers

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,
                              validation_data=val_generator, validation_steps=50,
                              verbose=1)


# We can see from the preceding output that our model has obtained a validation accuracy of around 96%,
# which is a 6% improvement from our previous model. Overall, this model has gained a 24% improvement in
# validation accuracy from our first basic CNN model. This really shows how useful transfer learning can be.
# We can see that accuracy values are really excellent here, and although the model looks like it might be
# slightly overfitting on the training data, we still get great validation accuracy. Let’s save this model to disk
# now using the following code.

model.save('cats_dogs_tlearn_finetune_img_aug_cnn.h5')


# Let’s now put all our models to the test by actually evaluating their performance on our test dataset.
#
# Evaluating our Deep Learning Models on Test Data
# We will now evaluate the five different models that we built so far, by first testing them on our test dataset,
# because just validation is not enough! We have also built a nifty utility module called model_evaluation_utils,
# which we will be using to evaluate the performance of our deep learning models. Let's load up the necessary
# dependencies and our saved models before getting started.

# load dependencies
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import model_evaluation_utils as meu

# %matplotlib inline

# load saved models
basic_cnn = load_model('cats_dogs_basic_cnn.h5')
img_aug_cnn = load_model('cats_dogs_cnn_img_aug.h5')
tl_cnn = load_model('cats_dogs_tlearn_basic_cnn.h5')
tl_img_aug_cnn = load_model('cats_dogs_tlearn_img_aug_cnn.h5')
tl_img_aug_finetune_cnn = load_model('cats_dogs_tlearn_finetune_img_aug_cnn.h5')

# load other configurations
IMG_DIM = (150, 150)
input_shape = (150, 150, 3)
num2class_label_transformer = lambda l: ['cat' if x == 0 else 'dog' for x in l]
class2num_label_transformer = lambda l: [0 if x == 'cat' else 1 for x in l]

# load VGG model for bottleneck features
from keras.applications import vgg16
from keras.models import Model
import keras

vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)
vgg_model.trainable = False


def get_bottleneck_features(model, input_imgs):
    features = model.predict(input_imgs, verbose=0)
    return features

# It’s time now for the final test, where we literally test the performance of our models by making predictions
# on our test dataset. Let’s load up and prepare our test dataset first before we try making predictions.
IMG_DIM = (150, 150)

test_files = glob.glob('test_data/*')
# test_files = glob.glob('kaggle_training/test/*')
test_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in test_files]
test_imgs = np.array(test_imgs)
test_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in test_files]

test_imgs_scaled = test_imgs.astype('float32')
test_imgs_scaled /= 255
test_labels_enc = class2num_label_transformer(test_labels)

print('Test dataset shape:', test_imgs.shape)
print(test_labels[975:980], test_labels_enc[0:5])


# Model 1: Basic CNN Performance
predictions = basic_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))

# Model 2: Basic CNN with Image Augmentation Performance
predictions = img_aug_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))

# Model 3: Transfer Learning — Pre-trained CNN as a Feature Extractor Performance
test_bottleneck_features = get_bottleneck_features(vgg_model, test_imgs_scaled)

predictions = tl_cnn.predict_classes(test_bottleneck_features, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))

# Model 4: Transfer Learning — Pre-trained CNN as a Feature Extractor with Image Augmentation Performance
predictions = tl_img_aug_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))

# Model 5: Transfer Learning — Pre-trained CNN with Fine-tuning and Image Augmentation Performance
predictions = tl_img_aug_finetune_cnn.predict_classes(test_imgs_scaled, verbose=0)
predictions = num2class_label_transformer(predictions)
meu.display_model_performance_metrics(true_labels=test_labels, predicted_labels=predictions,
                                      classes=list(set(test_labels)))

# worst model - basic CNN
meu.plot_model_roc_curve(basic_cnn, test_imgs_scaled,
                         true_labels=test_labels_enc,
                         class_names=[0, 1])

# best model - transfer learning with fine-tuning & image augmentation
meu.plot_model_roc_curve(tl_img_aug_finetune_cnn, test_imgs_scaled,
                         true_labels=test_labels_enc,
                         class_names=[0, 1])


# Case Study 2: Multi-Class Fine-grained Image Classification with Large Number of Classes and Less Data Availability

# Loading and Exploring the Dataset from https://www.kaggle.com/c/dog-breed-identification/data
# Let’s take a look at how our dataset looks like by loading the data and viewing a sample batch of images.
import scipy as sp
import numpy as np
import pandas as pd
import PIL
import scipy.ndimage as spi
import matplotlib.pyplot as plt

# % matplotlib inline
np.random.seed(42)

DATASET_PATH = r'./kaggle_training/train/'
LABEL_PATH = r'./kaggle_labels/labels.csv'


# This function prepares a random batch from the dataset
def load_batch(dataset_df, batch_size=25):
    batch_df = dataset_df.loc[np.random.permutation(np.arange(0,
                                                              len(dataset_df)))[:batch_size], :]
    return batch_df


# This function plots sample images in specified size and in defined grid
def plot_batch(images_df, grid_width, grid_height, im_scale_x, im_scale_y):
    f, ax = plt.subplots(grid_width, grid_height)
    f.set_size_inches(12, 12)

    img_idx = 0
    for i in range(0, grid_width):
        for j in range(0, grid_height):
            ax[i][j].axis('off')
            ax[i][j].set_title(images_df.iloc[img_idx]['breed'][:10])
            ax[i][j].imshow(sp.misc.imresize(spi.imread(DATASET_PATH + images_df.iloc[img_idx]['id'] + '.jpg'),
                                             (im_scale_x, im_scale_y)))
            img_idx += 1

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.25)


# load dataset and visualize sample data
dataset_df = pd.read_csv('kaggle_labels/labels.csv')
batch_df = load_batch(dataset_df, batch_size=36)
plot_batch(batch_df, grid_width=6, grid_height=6,
           im_scale_x=64, im_scale_y=64)

# Building Datasets
# Let’s start by looking at how the dataset labels look like to get an idea of what we are dealing with.

# data_labels = pd.read_csv('labels/labels.csv')
data_labels = pd.read_csv(LABEL_PATH)
target_labels = data_labels['breed']
print(len(set(target_labels)))
data_labels.head()

# What we do next is to add in the exact image path for each image present in the disk using the following code.
# This will help us in easily locating and loading up the images during model training.
train_folder = 'kaggle_training/train/'
data_labels['image_path'] = data_labels.apply(lambda row: (train_folder + row["id"] + ".jpg"),
                                              axis=1)
data_labels.head()


# It’s now time to prepare our train, test and validation datasets. We will leverage the following code
# to help us build these datasets!
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img

# load dataset
train_data = np.array([img_to_array(load_img(img, target_size=(299, 299)))
                       for img in data_labels['image_path'].values.tolist()
                       ]).astype('float32')

# create train and test datasets
x_train, x_test, y_train, y_test = train_test_split(train_data, target_labels,
                                                    test_size=0.3,
                                                    stratify=np.array(target_labels),
                                                    random_state=42)

# create train and validation datasets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.15,
                                                  stratify=np.array(y_train),
                                                  random_state=42)

print('Initial Dataset Size:', train_data.shape)
print('Initial Train and Test Datasets Size:', x_train.shape, x_test.shape)
print('Train and Validation Datasets Size:', x_train.shape, x_val.shape)
print('Train, Test and Validation Datasets Size:', x_train.shape, x_test.shape, x_val.shape)

# # We also need to convert the text class labels to one-hot encoded labels, else our model will not run.
# y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).as_matrix()
# y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).as_matrix()
# y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).as_matrix()

# We also need to convert the text class labels to one-hot encoded labels, else our model will not run.
y_train_ohe = pd.get_dummies(y_train.reset_index(drop=True)).values
y_val_ohe = pd.get_dummies(y_val.reset_index(drop=True)).values
y_test_ohe = pd.get_dummies(y_test.reset_index(drop=True)).values

y_train_ohe.shape, y_test_ohe.shape, y_val_ohe.shape


# Everything looks to be in order. Now, if you remember from the previous case study, image augmentation is a
# great way to deal with having less data per class. In this case, we have a total of 10222 samples and 120 classes.
# This means, an average of only 85 images per class! We do this using the ImageDataGenerator utility from keras.

from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32

# Create train generator.
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip='true')
train_generator = train_datagen.flow(x_train, y_train_ohe, shuffle=False,
                                     batch_size=BATCH_SIZE, seed=1)

# Create validation generator
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = train_datagen.flow(x_val, y_val_ohe, shuffle=False,
                                   batch_size=BATCH_SIZE, seed=1)

# Transfer Learning with Google’s Inception V3 Model
# Now that our datasets are ready, let’s get started with the modeling process. We already know how to build
# a deep convolutional network from scratch. We also understand the amount of fine-tuning required to achieve
# good performance. For this task, we will be utilizing concepts of transfer learning. A pre-trained model is the
# basic ingredient required to begin with the task of transfer learning.

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical

# Get the InceptionV3 model so we can do transfer learning
base_inception = InceptionV3(weights='imagenet', include_top=False,
                             input_shape=(299, 299, 3))

# Add a global spatial average pooling layer
out = base_inception.output
out = GlobalAveragePooling2D()(out)
out = Dense(512, activation='relu')(out)
out = Dense(512, activation='relu')(out)
total_classes = y_train_ohe.shape[1]
predictions = Dense(total_classes, activation='softmax')(out)

model = Model(inputs=base_inception.input, outputs=predictions)

# only if we want to freeze layers
for layer in base_inception.layers:
    layer.trainable = False

# Compile
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Based on the previous output, you can clearly see that the Inception V3 model is huge with a lot of layers and
# parameters. Let’s start training our model now. We train the model using the fit_generator(...) method to leverage
# the data augmentation prepared in the previous step. We set the batch size to 32, and train the model for 15 epochs.

# Train the model
batch_size = BATCH_SIZE
train_steps_per_epoch = x_train.shape[0] // batch_size
val_steps_per_epoch = x_val.shape[0] // batch_size

history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              validation_data=val_generator,
                              validation_steps=val_steps_per_epoch,
                              epochs=15, verbose=1)

# Evaluating our Deep Learning Model on Test Data
# Training and validation performance is pretty good, but how about performance on unseen data?
# Since we already divided our original dataset into three separate portions. The important thing to remember
# here is that the test dataset has to undergo similar pre-processing as the training dataset.
# To account for this, we scale the test dataset as well, before feeding it into the function.

# Added this code because there was an error because "labels_ohe_names" had not been defined
# but had been define later in the program...not sure why
labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
labels_ohe = np.asarray(labels_ohe_names)
label_dict = dict(enumerate(labels_ohe_names.columns.values))

# scaling test features
x_test /= 255.

# getting model predictions
test_predictions = model.predict(x_test)
predictions = pd.DataFrame(test_predictions, columns=labels_ohe_names.columns)
predictions = list(predictions.idxmax(axis=1))
test_labels = list(y_test)

# evaluate model performance
import model_evaluation_utils as meu
meu.get_metrics(true_labels=test_labels,
                predicted_labels=predictions)

# The model achieves an amazing 86% accuracy as well as F1-score on the test dataset. Given that we just trained
# for 15 epochs with minimal inputs from our side, transfer learning helped us achieve a pretty decent classifier.
# We can also check the per-class classification metrics using the following code.
meu.display_classification_report(true_labels=test_labels,
                                  predicted_labels=predictions,
                                  classes=list(labels_ohe_names.columns))

# We can also visualize model predictions in a visually appealing way using the following code.
grid_width = 5
grid_height = 5
f, ax = plt.subplots(grid_width, grid_height)
f.set_size_inches(15, 15)
batch_size = 25
dataset = x_test

# labels_ohe_names = pd.get_dummies(target_labels, sparse=True)
# labels_ohe = np.asarray(labels_ohe_names)
# label_dict = dict(enumerate(labels_ohe_names.columns.values))
model_input_shape = (1,)+model.get_input_shape_at(0)[1:]
random_batch_indx = np.random.permutation(np.arange(0,len(dataset)))[:batch_size]

img_idx = 0
for i in range(0, grid_width):
    for j in range(0, grid_height):
        actual_label = np.array(y_test)[random_batch_indx[img_idx]]
        prediction = model.predict(dataset[random_batch_indx[img_idx]].reshape(model_input_shape))[0]
        label_idx = np.argmax(prediction)
        predicted_label = label_dict.get(label_idx)
        conf = round(prediction[label_idx], 2)
        ax[i][j].axis('off')
        ax[i][j].set_title('Actual: '+actual_label+'\nPred: '+predicted_label + '\nConf: ' +str(conf))
        ax[i][j].imshow(dataset[random_batch_indx[img_idx]])
        img_idx += 1

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.5, hspace=0.55)