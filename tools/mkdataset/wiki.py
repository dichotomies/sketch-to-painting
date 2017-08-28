import h5py
import numpy as np
import operator
import os
import pandas as pd
import shutil
from   filedir import mk_dir


def load_table(path):
    store = pd.HDFStore(path, "r")
    return store["df"]


def filter_table(table, key, value, operator):
    """ 
    Filters table according to key-value pair and operator. Method can
    be concatenated for extracting subsets from original table.
    
    Examples
    ========
    filter_table(df, "artist_name", "a y jackson", operator.eq)
        filters in `df` entries where artists name is equal to "a y jackson"
        
    filter_table(df, "year", 1935, operator.ge)
        filters in `df` entries where paint year is greater than or equal 
        to 1935
    """
    indices = np.where(operator(table[key], value))[0]
    ids = table.image_id[indices]
    return table.loc[ids]


def merge_tables(*args):
    """ Merge tables that result from `filter_table`. """
    return pd.concat(args)


def save_images(table, path_i, path_o):
    """ Access images via IDs from table in `path_i` and save to `path_o`. """
    
    names = [name for name in table.image_id]
    
    mk_dir(path_o)

    for name in names:
        shutil.copyfile(os.path.join(path_i, name) + ".png", 
                os.path.join(path_o, name + ".png"))


if __name__ == "__main__":
    """ 
    Example usage: filter for "pablo picasso", then take entries where
    year of production is greater than or equal to 1920. Access images
    and save first 5 from the filtered table.
    """
    
    path_hdf = "/export/home/asanakoy/workspace/wikiart/info/info_v2.hdf5"
    path_i = "/export/home/asanakoy/workspace/wikiart/images_square_227x227"
    path_o = "pablo_i" # input for next step

    table = load_table(path_hdf)
    table = filter_table(table, "artist_name", "pablo picasso", operator.eq)
    table = filter_table(table, "year", 1920, operator.ge)
    
    save_images(table_b[:5], path_i, path_o)
