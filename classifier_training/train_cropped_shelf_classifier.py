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

TRAIN_DIR = "/hdd/datasets/DM/dataset_crop/train"
VAL_DIR = "/hdd/datasets/DM/dataset_crop/val"

IMG_HEIGHT = 168
IMG_WIDTH = 224

IMG_H_IN = 3000
IMG_W_IN = 4000

fx = 4696.604851 #/ 17.857142857142858
fy = 4704.904643 #/ 17.857142857142858
cx = 1998.531128 #/ 17.857142857142858
cy = 1446.352779 #/ 17.857142857142858

CLASSES = ["bucket", "hanging", "standing"]

# Select if mixup augmentation should be used
MIXUP = False
MIXUP_ALPHA = 0.2

# Path where model weights are saved
path_weights_best = "./best.h5"  # Early stopping
path_weights_last = "./last.h5"  # Model weights after last epoch


def prepare_images(path, set):
    img_list = []
    index = 0
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))

        if img is not None:
            stats = filename.split("_")
            x = float(stats[2])
            y = float(stats[4])
            z = float(stats[6])
            if y < 0.0 and stats[0] == 'bucket':
                continue
            u = (fx * x) / z
            v = (fy * y) / z
            hwin_x = (fx * 0.08) / z
            hwin_y = (fy * 0.06) / z

            img = np.pad(img, ((1000, 1000), (1000, 1000), (0,0)), 'edge')
            img = img[int(1000 + cy + v - hwin_y):int(1000+ cy + v + hwin_y), int(1000 + cx + u - hwin_x):int(1000 + cx + u + hwin_x), :]
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            rnd_name = 'img_' + str(index) + '.png'
            viz_path = '/hdd/datasets/DM/dataset_crop/' + set + '/' + str(stats[0]) + '/' + rnd_name
            print(viz_path)
            cv2.imwrite(viz_path, img)

            index += 1

    return np.array(img_list)


def load_images(path):
    img_list = []

    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_list.append(img_rgb)
    return np.array(img_list)


# Load validation and train images and crop based on barcode
#prepare_images(VAL_DIR + "/bucket", set='val')
#prepare_images(VAL_DIR + "/hanging", set='val')
#prepare_images(VAL_DIR + "/standing", set='val')

#prepare_images(TRAIN_DIR + "/bucket", set='train')
#prepare_images(TRAIN_DIR + "/hanging", set='train')
#prepare_images(TRAIN_DIR + "/standing", set='train')

# Load validation and test datasets for classification report
x_bucket_val = load_images(VAL_DIR + "/bucket").astype("float32") / 255.0
x_hanging_val = load_images(VAL_DIR + "/hanging").astype("float32") / 255.0
x_standing_val = load_images(VAL_DIR + "/standing").astype("float32") / 255.0

n_bucket_val = x_bucket_val.shape[0]
n_hanging_val = x_hanging_val.shape[0]
n_standing_val = x_standing_val.shape[0]

y_bucket_val = np.zeros((n_bucket_val, 3), dtype="float32")
y_bucket_val[:, 0] = 1
y_hanging_val = np.zeros((n_hanging_val, 3), dtype="float32")
y_hanging_val[:, 1] = 1
y_standing_val = np.zeros((n_standing_val, 3), dtype="float32")
y_standing_val[:, 2] = 1


# Select parameters for augmenatation here (except mixup))
train_imgen = K.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=False,
    fill_mode='nearest')

val_imgen = K.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)


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

print("Validation Dataset (best model:")
y_val_true = np.argmax(np.vstack((y_bucket_val, y_hanging_val, y_standing_val)), axis=1)
x_val = np.vstack((x_bucket_val, x_hanging_val, x_standing_val))
y_val_pred = np.argmax(best_model.predict(x_val), axis=1)
print(classification_report(y_val_true, y_val_pred))