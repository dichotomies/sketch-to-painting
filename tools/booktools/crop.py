import argparse
import os
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing
from skimage.morphology import binary_opening
from skimage.morphology import remove_small_objects
from skimage.morphology import convex_hull_image
import numpy as np
from skimage.io import imsave, imread, imshow
from skimage import img_as_ubyte
from skimage.color import rgb2gray


# constants for margins of book (around stripes around actual image)
PAGE_STRIPE_H = 30
PAGE_STRIPE_W  = 30


def mk_dir(path):
    if not os.path.exists(path):
        print('Creating output directory', path)
        os.makedirs(path)


def read_im(path, gray = False, binarize = False):

    im = imread(path)
    im = img_as_ubyte(im)
    if gray == True: im = rgb2gray(im)
    if binarize == True: im = (im > 128).astype("bool")
    return im


def get_lrtb(im, nb_cc = 1024):

    kernel = np.ones((15, 15), np.bool)
    closing = binary_closing(im, selem = kernel)
    opening = binary_opening(closing, selem = kernel)

    clean = remove_small_objects(opening.astype('bool'), min_size = nb_cc)

    imshow(clean)

    hull = convex_hull_image(clean)
    
    # get left, right, top, bottom indices
    hull_indices = np.array(np.where(hull == 1))
    l = np.min(hull_indices[0]) + PAGE_STRIPE_W
    r = np.max(hull_indices[0]) - PAGE_STRIPE_W
    t = np.min(hull_indices[1]) + PAGE_STRIPE_H
    b = np.max(hull_indices[1]) - PAGE_STRIPE_H
    return l, r, t, b


def crop(im, l, r, t, b):
    aoi = im[l:r, t:b]
    aoi_uint = img_as_ubyte(aoi, force_copy = True)
    # aoi_uint[aoi == 0] = 255
    # aoi_uint[aoi == 1] = 0

    return aoi_uint


def plot_im(im):

    plt.figure(figsize = (20, 10))
    imshow(im)
    plt.show()


def transform_save_cropped(path_color, path_edges, path_cropped, 
        nb_connected_components = 10000, nb_max_images = 0):

    # import warnings
    # warnings.filterwarnings("ignore")

    mk_dir(os.path.join(path_cropped, 'color'))
    mk_dir(os.path.join(path_cropped, 'edges'))

    if nb_max_images == 0:
        im_names = os.listdir(path_edges)
    else:
        im_names = os.listdir(path_edges)[:nb_max_images]

    for i, name in enumerate(im_names):
        
        if i % 10 == 0:
            print("Processing image", i + 1, "out of", len(im_names))

        im_binary = read_im(os.path.join(path_edges, name), True, True)
        
        try:
            
            l, r, t, b = get_lrtb(im_binary, nb_connected_components)

            im_color = read_im(os.path.join(path_color, name), False, False)
            im_color_cropped = im_color[l: r, t : b]
            im_edges = read_im(os.path.join(path_edges, name), True, False)
            im_edges_cropped = im_edges[l:r, t:b]
            imsave(os.path.join(path_cropped, 'color', name), im_color_cropped)
            imsave(os.path.join(path_cropped, 'edges', name), im_edges_cropped)

        except:

            print("Error occured for image with name", name)
            print("Try a different amount of connected components.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Crop margins from pages in'+
            ' book.')
    parser.add_argument('--path_color', dest='path_color', type=str,
            action='store', default='color', help='path to color images ' +
            'that will be cropped')
    parser.add_argument('--path_edges', dest='path_edges', type=str,
            action='store', default='edges', help='path to edge images that '+
            'will be used for extracting connected components ' + 
            'and that will be cropped')
    parser.add_argument('--path_cropped', dest='path_cropped', type=str,
            action='store', default='cropped', help='output path for ' + 
            'cropped images')
    parser.add_argument('--nb_connected_components', dest='nb_cc', type=int,
            action='store', default=10000, help='amount of connected ' + 
            'components used for cropping the margins')
    parser.add_argument('--nb_convert_images', dest='nb_ims', type=int,
            action='store', default=0, help='maximum number of images that ' + 
            'will be cropped')

    args = parser.parse_args()
    transform_save_cropped(args.path_color, args.path_edges, 
            args.path_cropped, args.nb_cc, args.nb_ims)
