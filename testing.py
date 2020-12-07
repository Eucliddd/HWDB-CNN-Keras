import getopt
import os
import sys

import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array

number_of_classes = 3755
save_path = "trained_best_weights_1.h5"
_label = [dir_name for dir_name in os.listdir("D:\\validation")]
number_of_classes = len(_label)


def read_test(PATH):
    BS = 32
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    testGen = test_datagen.flow_from_directory(PATH,
                                               target_size=(64, 64),
                                               batch_size=BS,
                                               classes=_label,
                                               class_mode="categorical",
                                               color_mode="grayscale")
    STEP_VALIDATION = testGen.n // testGen.batch_size
    model = load_model(save_path)
    score = model.evaluate_generator(testGen, steps=STEP_VALIDATION)
    print("样本准确率%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))


def read_image(PATH):
    I = Image.open(PATH).convert('L').resize((64, 64))
    Idarry = img_to_array(I, dtype='float32')
    Idarry = Idarry / 255
    model = load_model(save_path)
    Idarry = np.expand_dims(Idarry, axis=0)
    preds = model.predict(Idarry)
    print("Label is :", _label[np.argmax(preds)])


if __name__ == '__main__':
    file_path = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:f:", ["predict=", "file_path="])
        print(args, opts)
    except getopt.GetoptError:
        print('Error: testing.py -m <''image'' or ''folder''> -f <file path>')
        print('   or: testing.py --predict=<''image'' or ''folder''> --file_path=<file path>')
        sys.exit(2)
if opts[0][1] == "image":
    read_image(opts[1][1])
if opts[0][1] == "folder":
    read_test(opts[1][1])
else:
    print("the first argument should be ''image'' or ''folder''")
