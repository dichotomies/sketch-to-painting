

import argparse
import matplotlib.pyplot as plt
import os
from skimage import feature
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.io import imread, imsave, imshow
from skimage.util import invert
import sys


def save_binarized_sketches(path_i, path_o, nb_ims = 0):
    if os.path.isdir(path_o) != True:
        os.makedirs(path_o)

    names = [n for n in os.listdir(path_i) if n[-4:] == ".png"]

    if nb_ims != 0: names = names[:nb_ims]

    for i, name in enumerate(names):
        
        if i > 0 and i % 10 == 0:
            print str(i) + "/" + str(len(names))
        im = imread(os.path.join(path_i, name))
        im = rgb2gray(im)
        edges = feature.canny(im, sigma = 2.5)
        filepath = os.path.join(path_o, name[:-4] + ".png")
        edges = img_as_ubyte(edges)
        imsave(filepath, invert(edges))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Binarize sketches from book')
    parser.add_argument('--path_sketch', dest='path_i', type=str,
            action='store', help='path to sketches ' +
            'that will be binarized')
    parser.add_argument('--path_binary', dest='path_o', type=str,
            action='store', help='output path for ' + 
            'binarized images')
    parser.add_argument('--nb_convert_images', dest='nb_ims', type=int,
            action='store', default=0, help='maximum number of images that ' + 
            'will be binarized')

    args = parser.parse_args()
    save_binarized_sketches(args.path_i, args.path_o, args.nb_ims)
