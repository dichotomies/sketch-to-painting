# Prerequisities and General Notes

Python version: 2.

Code was tested with newest versions of packages (see `required.txt` for list 
of packages that are required).

Install Caffe-HED-version according to https://github.com/s9xie/hed. Make sure
to add caffe to Python path, e.g. add something like following lines to bashrc,
zshrc etc.:

```
# necessary for caffe
export PYTHONPATH=$HOME/.local/share/caffe/python:$PYTHONPATH
export CAFFE_ROOT=$HOME/.local/share/caffe
```

This manual is based on pictures extracted from `wiki` for the artist Picasso.
Alternatives are `hardpaint` and `book`.

Main scripts have command line arguments, get help with `python <python 
script file name> --help`

# Tools for Image Processing and Dataset Creation

## Crop Margins (`booktools/crop.py`)

This package is only necessary if images of the book are used as input for
synthim, i.e. the margins and stripes of the single pages must be removed. It
can also be used, if the task is to crop the margins from the sketches.

First, create edges (necessary for input, i.e. generation of closing, opening
and connected components) with HED from synthim (see above).

Finally, crop margin of color and edge images and use the cropped folder for
synthim:

```
python crop.py \
--path_color /export/home/vtschern/workspace/_images/book/color \
--path_edges /export/home/vtschern/workspace/_images/book/edges \
--path_cropped /export/home/vtschern/workspace/_images/book/cropped \
--nb_convert_images 5 --nb_connected_components 10000
```

Here, the folder `cropped` contains `color` and `edges`. Both folders would
be used for the next steps like the other pictures (point towards these 
folders if asked for `--path_color` and `--path_edges`).


## Extract Edges from Sketches with Canny (`booktools/binarize.py`)

```
python binarize.py \
--path_sketch /export/home/vtschern/data/book-original-images/\
pages-valid/sketch \
--path_binary /export/home/vtschern/workspace/_images/book/sketch2 \
--nb_convert_images 5
```

## Extract Images from Wiki-HDF (`mkdataset/wiki.py`)

This module must imported in a script if used, since functionality of passing parameters via CL is limited.

An example script can be found in `scripts`.

It is possible to specify operators and filter the wiki pandas dataframe for key-value pairs, e.g.:

```python
table = load_table(path_hdf)
table = filter_table(table, "artist_name", "pablo picasso", operator.eq)
table = filter_table(table, "year", 1920, operator.ge)
```

This script filters first for artists named Pablo Picasso ("operator.equal"), then it filters the result for any paintings created in 1920 or after 1920 ("operator.greaterequal").

## Generating Edges from Images (`mkdataset/edges.py`)

This module generates edges (output) from colored images (input) with HED.

```
CUDA_VISIBLE_DEVICES=3 python edges.py \
--path_input /export/home/vtschern/workspace/_images/picasso/color \
--path_output /export/home/vtschern/workspace/_images/picasso/edges \
--path_hed /export/home/vtschern/workspace/mkdataset/configs/hed \
--nb_convert_images 5
```

## Generating Datasets (`mkdataset/dataset.py`)

```
python dataset.py \
--path_color /export/home/vtschern/workspace/_images/picasso/color \
--path_edges /export/home/vtschern/workspace/_images/picasso/edges \
--path_dataset /export/home/vtschern/workspace/synthim/_datasets/picasso \
--nb_convert_images 5
```


# Synthesize Images with (`synthim`)

Images can have arbitrary size, because they are downscaled and cropped to
input size of the adversarial net if necessary.

Datasets are assumed to be in main directory of `synthim` and named `_datasets`.

Pretrained models can be used for testing and synthesizing. Options for 
`--model` are: `picasso`, `book`, `hardpaint`. Synthesizing can be done with
sketches are edges (i.e., no dataset is needed). 

Further options can be found in the folder `options`.

## Training a Model

```
CUDA_VISIBLE_DEVICES=3 python train.py \
--dataroot ./_datasets/picasso \
--name picasso --dataset_mode aligned --no_lsgan --niter 900 
```

- `--name` describes model name (also used as name for testing the model).
- `--no_lsgan` chooses vanilla GAN instead of least-squares GAN

Realtime visualization of training and testing results can be enabled with
visdom server: `--display_id <1, 2, ...>`. Select port for
displaying with `--display-port <port number>`, otherwise 8097 is chosen.
Connect to visdom server via `localhost:<port>` in web browser.

Number of iterations includes amount of iterations for decaying the learning
rate linearly to zero: `--niter_decay`. Its default value is 100. Non-decaying
iterations can be added with `--niter <100, 200, ...>`; default value being
100.

## Testing a Model

- i.e., use only data from test set (data that was not used while training)

The amount of pictures synthesized is limited to 50, adapt with the option 
`--how_many`.

```
CUDA_VISIBLE_DEVICES=3 python test.py \
--dataroot ./_datasets/picasso \
--name picasso --dataset_mode aligned
```

## Synthesizing Images

The amount of pictures synthesized is limited to 50, adapt with the option 
`--how_many`.

For example, generating images from machine-generated edges:

```
CUDA_VISIBLE_DEVICES=3 python synth.py \
--dataroot /export/home/vtschern/workspace/_images/picasso/edges \
--name picasso
```

For example, generating images from human-drawn sketches:

```
CUDA_VISIBLE_DEVICES=3 python synth.py \
--dataroot /export/home/vtschern/workspace/_images/picasso-sketch/binary/ \
--name picasso
```

