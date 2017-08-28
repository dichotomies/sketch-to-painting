import argparse
import os
import numpy as np
from skimage.io import imread, imsave, imshow
import skimage
from sklearn.utils import shuffle
from filedir import mk_dir


# ratio for train and test set
RATIO = (0.7, 0.3)


def save_as_dataset(path_color, path_edges, path_dataset, nb_ims = 0, 
        tr_te_ratio = RATIO):
    
    mk_dir(os.path.join(path_dataset, "train"))
    mk_dir(os.path.join(path_dataset, "test"))

    names = []
    for i in [j for j in os.listdir(path_color)]:
        names.append(i)

    if nb_ims > 0:
        names = names[:nb_ims]
        
    assert RATIO[0] + RATIO[1] == 1.0

    im_names = os.listdir(path_color)

    # shuffle for training and test set
    im_names = shuffle(im_names)

    size_tr = round(len(im_names) * RATIO[0])

    for i, name in enumerate(names):

        if i > 0 and i % 20 == 0:
            print('processing image %d/%d' % (i, len(names)))

        l = imread(os.path.join(path_color, name))
        r = imread(os.path.join(path_edges, name))

        r = skimage.img_as_ubyte(r)
        r = skimage.color.gray2rgb(r)
        a = np.zeros((l.shape[0], (l.shape[1]) * 2, 3)).astype("uint8")
        a[:, :l.shape[1]] = l
        a[:, l.shape[1] : (l.shape[1]) * 2] = r

        name_dir = "train" if (i < size_tr) else "test"

        imsave(os.path.join(path_dataset, name_dir, name), a)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create datasets for model')
    parser.add_argument('--path_color', dest='path_color', type=str,
            action='store', help='path to color images')
    parser.add_argument('--path_edges', dest='path_edges', type=str,
            action='store', help='path to edge images')
    parser.add_argument('--path_dataset', dest='path_dataset', type=str,
            action='store', help='path to dataset folder (output)')
    parser.add_argument('--nb_convert_images', dest='nb_ims', type=int,
            action='store', default=0, help='maximum number of images that ' +
            'will be converted')

    args = parser.parse_args()
    save_as_dataset(args.path_color, args.path_edges, args.path_dataset, 
            args.nb_ims)
