import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

matplotlib.use("Agg")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def GNT_generator(path, bs, start, end, mode='train', aug=None):
    idx = start - 1
    # print(start, "a epoch")
    while True:
        labels = []
        images = []
        # count = 0
        # print(idx, "start")
        for count in range(0, bs):
            try:
                # count = count + 1
                idx = idx + 1
                # print(count, idx)
                if idx == end:
                    # print("reach end")
                    idx = start
                    if mode == 'eval':
                        break
                for dir_name in _label:
                    root_path = os.path.join(path, dir_name)
                    filename = os.path.join(root_path, str(idx) + '.png')
                    try:
                        im = Image.open(filename).resize([64, 64])
                    except IOError:
                        # print("Error: no file name %s" % filename)
                        continue
                    image = np.asarray(im).reshape((64, 64, 3))
                    label = dir_name
                    images.append(image)
                    labels.append(label)
            except StopIteration:
                # print("ERROR at", count, idx)
                continue
        # print(" a batchsize")
        labels = lb.transform(np.array(labels))
        images = np.array(images)
        if aug is not None:
            (images, labels) = next(aug.flow(images,
                                             labels, batch_size=bs))

        yield (images, labels)


TRAIN_PATH = "D:\\train"
TEST_PATH = "D:\\test"
BS = 32
_label = [dir_name for dir_name in os.listdir(TRAIN_PATH)]
number_of_classes = len(_label)
num_of_epochs = 12
STEP_TRAIN = 60
STEP_VALIDATION = 60


def build_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3),
                     activation='relu', padding='same', strides=(1, 1),
                     input_shape=(64, 64, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3),
                     activation='relu', padding='same', strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3),
                     activation='relu', padding='same', strides=(1, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))

    # model.add(Conv2D(128, (3, 3), input_shape=(64, 64, 3)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, (3, 3)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, (3, 3)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Flatten())
    #
    # # Fully connected layer
    # model.add(Dense(1024))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(number_of_classes))
    #
    # model.add(Activation('softmax'))
    return model


save_path = "D:\\CNN\\trained_best_weights_1.h5"


def training(train, test):
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(save_path,
                                 monitor='val_loss', verbose=1, save_best_only=True, period=1)

    tensor_board = TensorBoard(log_dir='/content/drive/My Drive/CNN', write_graph=True, batch_size=BS)
    model.summary()
    if os.path.exists(save_path):
        model = load_model(save_path)
        # 若成功加载前面保存的参数，输出下列信息
        print("checkpoint_loaded")

    H = model.fit(
        train,
        steps_per_epoch=STEP_TRAIN,
        validation_data=test,
        validation_steps=STEP_VALIDATION,
        verbose=2,
        epochs=num_of_epochs,
        callbacks=[checkpoint, tensor_board]
    )  # the more epoch the better
    model.save('model.h5')
    # test = GNT_generator(TEST_PATH, BS, 193, 241, mode='eval', aug=None)

    N = num_of_epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")


def testing(test):
    # load model
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(save_path)
    predIdxs = model.predict_generator(test,
                                       steps=(STEP_VALIDATION) + 1)
    predIdxs = np.argmax(predIdxs, axis=1)
    print("[INFO] evaluating network...")
    print(classification_report(test.classes, predIdxs,
                                target_names=lb.classes_))
    print(model.metrics)


if __name__ == "__main__":
    lb = LabelBinarizer()
    lb.fit(_label)
    mlabels = lb.transform(_label)
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=0,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)
    trainGen = train_datagen.flow_from_directory(TRAIN_PATH,
                                                 target_size=(64, 64),
                                                 batch_size=BS,
                                                 classes=_label,
                                                 class_mode="categorical",
                                                 color_mode="grayscale")

    STEP_TRAIN = trainGen.n // trainGen.batch_size
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    testGen = test_datagen.flow_from_directory(TEST_PATH,
                                               target_size=(64, 64),
                                               batch_size=BS,
                                               classes=_label,
                                               class_mode="categorical",
                                               color_mode="grayscale")
    STEP_VALIDATION = testGen.n // testGen.batch_size
    training(trainGen, testGen)
    testing(testGen)
