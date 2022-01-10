import tensorflow.keras as K
from tensorflow.keras.applications import resnet_v2
import numpy as np
import os
from sklearn.metrics import classification_report
import cv2
from mixup_generator import MixupImageDataGenerator


BATCH_SIZE = 32
EPOCHS = 50


# Specify path of datasets for training and validation
TRAIN_DIR = "./images_segmentation_augmented/train"
VAL_DIR = "./images_segmentation_augmented/val"
TEST_DIR = "./images_segmentation_augmented/test"

IMG_HEIGHT = 168
IMG_WIDTH = 224

CLASSES = ["bucket", "hanging", "standing"]

# Select if mixup augmentation should be used
MIXUP = True
MIXUP_ALPHA = 0.2

# Path where model weights are saved
path_weights_best = "./best.h5"  # Early stopping
path_weights_last = "./last.h5"  # Model weights after last epoch


def load_images(path):
    img_list = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img_rgb)
    return np.array(img_list)


# Load validation and test datasets for classification report
x_bucket_val = load_images(VAL_DIR + "/bucket").astype("float32") / 255.0
x_hanging_val = load_images(VAL_DIR + "/hanging").astype("float32") / 255.0
x_standing_val = load_images(VAL_DIR + "/standing").astype("float32") / 255.0

x_bucket_test = load_images(TEST_DIR + "/bucket").astype("float32") / 255.0
x_hanging_test = load_images(TEST_DIR + "/hanging").astype("float32") / 255.0
x_standing_test = load_images(TEST_DIR + "/standing").astype("float32") / 255.0

n_bucket_val = x_bucket_val.shape[0]
n_hanging_val = x_hanging_val.shape[0]
n_standing_val = x_standing_val.shape[0]

n_bucket_test = x_bucket_test.shape[0]
n_hanging_test = x_hanging_test.shape[0]
n_standing_test = x_standing_test.shape[0]

y_bucket_val = np.zeros((n_bucket_val, 3), dtype="float32")
y_bucket_val[:, 0] = 1
y_hanging_val = np.zeros((n_hanging_val, 3), dtype="float32")
y_hanging_val[:, 1] = 1
y_standing_val = np.zeros((n_standing_val, 3), dtype="float32")
y_standing_val[:, 2] = 1

y_bucket_test = np.zeros((n_bucket_test, 3), dtype="float32")
y_bucket_test[:, 0] = 1
y_hanging_test = np.zeros((n_hanging_test, 3), dtype="float32")
y_hanging_test[:, 1] = 1
y_standing_test = np.zeros((n_standing_test, 3), dtype="float32")
y_standing_test[:, 2] = 1



# Select parameters for augmenatation here (except mixup))
train_imgen = K.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    brightness_range=(0.9, 1.1),
    horizontal_flip=True,
    fill_mode='nearest')

val_imgen = K.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)

# Create training and validation generator.
if MIXUP:
    train_generator = MixupImageDataGenerator(generator=train_imgen,
                                              directory=TRAIN_DIR,
                                              classes=CLASSES,
                                              batch_size=BATCH_SIZE,
                                              img_height=IMG_HEIGHT,
                                              img_width=IMG_WIDTH,
                                              alpha=MIXUP_ALPHA)
else:
    train_generator = train_imgen.flow_from_directory(directory=TRAIN_DIR,
                                                      classes=CLASSES,
                                                      target_size=(
                                                          IMG_HEIGHT, IMG_WIDTH),
                                                      class_mode="categorical",
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True)

validation_generator = val_imgen.flow_from_directory(directory=VAL_DIR,
                                                     classes=CLASSES,
                                                     target_size=(
                                                         IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode="categorical",
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True)

# Create NN for classification
resnet_input = K.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
resnet_conv = resnet_v2.ResNet50V2(include_top=False, input_tensor=resnet_input, weights=None,
                                   input_shape=resnet_input[0].get_shape(), classes=len(CLASSES), pooling='avg')
resnet_predictions = K.layers.Dense(len(CLASSES), activation='softmax')(resnet_conv.output)
resnet_model = K.Model(inputs=resnet_input, outputs=resnet_predictions)

resnet_model.compile(optimizer=K.optimizers.Adam(learning_rate=0.0005), loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Callback for early stopping
safe_best = K.callbacks.ModelCheckpoint(filepath=path_weights_best, monitor='val_accuracy', mode='max',
                                        save_best_only=True)
# Callback for learn rate reduction
red_learn_rate = K.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.1,
    patience=5,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
)

if MIXUP:
    steps_per_epoch = train_generator.get_steps_per_epoch()
else:
    steps_per_epoch = train_generator.samples // BATCH_SIZE

# Start training of model
history_resnet = resnet_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[safe_best, red_learn_rate])

# Save last model weights
resnet_model.save(path_weights_last)

# Load early stopping weights for classification report
best_model = K.models.load_model(path_weights_best)

# Print classification report of test and val dataset
print("Validation Dataset (last model:")
y_val_true = np.argmax(np.vstack((y_bucket_val, y_hanging_val, y_standing_val)), axis=1)
x_val = np.vstack((x_bucket_val, x_hanging_val, x_standing_val))
y_val_pred = np.argmax(resnet_model.predict(x_val), axis=1)
print(classification_report(y_val_true, y_val_pred))

print("Test Dataset (last model):")
y_test_true = np.argmax(np.vstack((y_bucket_test, y_hanging_test, y_standing_test)), axis=1)
x_test = np.vstack((x_bucket_test, x_hanging_test, x_standing_test))
y_test_pred = np.argmax(resnet_model.predict(x_test), axis=1)
print(classification_report(y_test_true, y_test_pred))

print("Validation Dataset (best model:")
y_val_true = np.argmax(np.vstack((y_bucket_val, y_hanging_val, y_standing_val)), axis=1)
x_val = np.vstack((x_bucket_val, x_hanging_val, x_standing_val))
y_val_pred = np.argmax(best_model.predict(x_val), axis=1)
print(classification_report(y_val_true, y_val_pred))

print("Test Dataset (best model):")
y_test_true = np.argmax(np.vstack((y_bucket_test, y_hanging_test, y_standing_test)), axis=1)
x_test = np.vstack((x_bucket_test, x_hanging_test, x_standing_test))
y_test_pred = np.argmax(best_model.predict(x_test), axis=1)
print(classification_report(y_test_true, y_test_pred))
