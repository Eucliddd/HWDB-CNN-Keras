import os
import struct

from PIL import Image


def extract_imgs(src):
    """extract image object from gnt file;\n
       src: the file of gnt file
    """
    res = {}
    count = 0
    f = open(src, 'rb')
    while f.read(1) != "":
        f.seek(-1, 1)
        count += 1
        Data = f.read(4)
        try:
            struct.unpack('<I', Data)
        except:
            break
        tag_code = f.read(2)
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]

        im = Image.new('RGB', (width, height))
        img_array = im.load()

        for x in range(0, height):
            for y in range(0, width):
                pixel = struct.unpack('<B', f.read(1))[0]
                img_array[y, x] = (pixel, pixel, pixel)

        tag = tag_code.decode("gbk")

        if tag in res:
            print("more code data")
        else:
            res[tag] = im

    f.close()
    return res


def save_imgs(word, im, path):
    '''save image object in file;\n
       word: the chinese character;\n
       im: the
    '''
    dirpath = os.path.join(path, word)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    for i, img in enumerate(im):
        filename = str(i) + '.png'
        img.save(os.path.join(dirpath, str(i) + ".png"))


if __name__ == "__main__":
    path = "D:\\HWDB"
    trainpath = "D:\\validation"
    testpath = "D:\\test"
    for i in range(241, 301):
        fname = os.path.join(path, "1%s-c.gnt" % (str(i).zfill(3)))
        sample = extract_imgs(fname)
        print("%s finished" % fname)
        for k in sample:
            dirpath = os.path.join(trainpath, k)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            filename = str(i) + '.png'
            sample[k].save(os.path.join(dirpath, str(i) + ".png"))
    # for i in range(193, 241):
    #     fname = os.path.join(path, "1%s-c.gnt" % (str(i).zfill(3)))
    #     sample = extract_imgs(fname)
    #     print("%s finished" % fname)
    #     for k in sample:
    #         dirpath = os.path.join(testpath, k)
    #         if not os.path.exists(dirpath):
    #             os.makedirs(dirpath)
    #         filename = str(i) + '.png'
    #         sample[k].save(os.path.join(dirpath, str(i) + ".png"))
