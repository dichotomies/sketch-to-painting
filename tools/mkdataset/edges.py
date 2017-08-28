"""
    Transform original images into images containing only edges.  
"""


import argparse
import numpy as np
import os
from   skimage.util import invert
from   skimage      import img_as_float
from   skimage.io   import imsave, imread
import sys
from   filedir import mk_dir


# --- related to caffe
# set output to `only warnings`, such that no model info is shown
os.environ['GLOG_minloglevel'] = '2' 
# caffe must be set in env. variable for python path
import caffe  


PATH_HED = "/export/home/vtschern/workspace/mkdataset/configs/hed"
PATH_PROTO = os.path.join(PATH_HED, 'deploy.prototxt')
PATH_MODEL = os.path.join(PATH_HED, 'hed_pretrained_bsds.caffemodel')
SZ_BORDER = 128
H = 35
W = 35

def save_as_edges(path_i, path_o, nb_ims_max = 0):

    mk_dir(path_o)

    im_names = os.listdir(path_i)
    nb_ims = len(im_names)
    if nb_ims_max == 0: nb_ims_max = nb_ims
    print('#images = %d' % nb_ims)

    # load net
    caffe.set_mode_gpu()

    for i in range(nb_ims)[:nb_ims_max]:
    
        net = caffe.Net(PATH_PROTO, PATH_MODEL, caffe.TEST)
        
        if i > 0 and i % 20 == 0:
            print('processing image %d/%d' % (i, nb_ims))
        im = imread(os.path.join(path_i, im_names[i]))

        im = np.array(im, dtype=np.float32)
        im = np.pad(im,((SZ_BORDER, SZ_BORDER),(SZ_BORDER,SZ_BORDER),(0,0)),
               'reflect')

        im = im[:,:,::-1]
        im = im.transpose((2, 0, 1))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *im.shape)
        net.blobs['data'].data[...] = im
        
        # run net and take argmax for prediction
        net.forward()
        pred = net.blobs['sigmoid-fuse'].data[0][0, :, :]
        
        # get rid of the border
        # pred = pred[SZ_BORDER : -SZ_BORDER, SZ_BORDER : -SZ_BORDER]
        pred = pred[SZ_BORDER + W : - SZ_BORDER + W, 
                SZ_BORDER + H : - SZ_BORDER + H]
        
        # save image
        name, ext = os.path.splitext(im_names[i])
        out = img_as_float(pred)
        out = invert(out)
        imsave(os.path.join(path_o, name + '.png'), out)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Extract edges with HED.')
    parser.add_argument('--path_input', dest='path_i', type=str,
            action='store', help='path to images that will be converted ' +
            'into edges')
    parser.add_argument('--path_output', dest='path_o', type=str,
            action='store', help='output path to edge images')
    parser.add_argument('--path_hed', dest='path_hed', type=str,
            action='store', default='configs/hed', help='path to location ' +
            'HED config files: deploy.prototext and ' +
            'hed_pretrained_bsds.caffemodel')
    parser.add_argument('--nb_convert_images', dest='nb_ims', type=int,
            action='store', default=0, help='maximum number of images that ' +
            'will be converted')

    args = parser.parse_args()
    PATH_HED = args.path_hed
    save_as_edges(args.path_i, args.path_o, args.nb_ims)
