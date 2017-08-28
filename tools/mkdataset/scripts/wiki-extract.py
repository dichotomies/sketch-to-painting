""" 
    Extract first five images for paintings of picasso and paintings with 
    hardpainting style.
"""

# import os
# os.chdir('/export/home/vtschern/workspace/mkdataset')

import sys
sys.path.insert(0, "../")

from wiki import load_table, filter_table, save_images
import operator


path_hdf = "/export/home/asanakoy/workspace/wikiart/info/info_v2.hdf5"
path_i = "/export/home/asanakoy/workspace/wikiart/images_square_227x227"


# pictures of pablo picasso

path_o = "picasso-color" # input for next step

table = load_table(path_hdf)
table = filter_table(table, "artist_name", "pablo picasso", operator.eq)
table = filter_table(table, "year", 1920, operator.ge)

save_images(table[:], path_i, path_o)


# pictures with style `hard paintings`

path_o = "hp-color" # input for next step

table = load_table(path_hdf)
table = filter_table(table, "style", "hard edge painting", operator.eq)

save_images(table[:], path_i, path_o)

