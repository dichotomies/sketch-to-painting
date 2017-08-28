

import os

def mk_dir(path):

    if not os.path.exists(path):
        print('Creating output directory %s' % path)
        os.makedirs(path)


