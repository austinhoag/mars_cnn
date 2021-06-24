import os,glob
import concurrent.futures
from PIL import Image
import pandas as pd
import numpy as np
import shutil

input_dir = '/home/ahoag/ML/mars/gallery_app/app/static/images'
trainval_dir = '/home/ahoag/ML/mars/craters_dunes/images_for_cnn/neither'
test_dir = '/home/ahoag/ML/mars/craters_dunes/test_cnn/neither'

def make_thumb(filename,dirname):
    """
    ---PURPOSE---
    make a 50x50 thumbnail of all images labeled as having a crater 
    and save to static/images/craters folder
    ---INPUT---
    filename - referring to a 200x200 size chunked jpeg
    dirname  - directory in which to save the thumbs
    """
    output_size=(180,180)
    basename = os.path.basename(filename)
    savename_thumb = os.path.join(dirname,basename)
    
    with Image.open(filename) as pil_im:
        assert pil_im.mode == 'RGB'
        assert pil_im.size == (200,200)
        pil_im.thumbnail(output_size)
        pil_im.save(savename_thumb)

if __name__=='__main__':
    neither_filenames = glob.glob(input_dir + '/neither_dune_nor_crater/*jpg')    
    np.random.seed(42)
    np.random.shuffle(neither_filenames)
    neither_filenames_train_val = neither_filenames[0:750]
    neither_filenames_test = neither_filenames[750:]

    """ Train/val set first """
    for ii,full_filename in enumerate(neither_filenames_train_val):
        if ii % 10 == 0:
            print(ii+1,'/',len(neither_filenames_train_val))
        basename = os.path.basename(full_filename)
        dst_filename = os.path.join(trainval_dir,basename)
        shutil.copyfile(full_filename,dst_filename)
    """ Now test set """
    for ii,full_filename in enumerate(neither_filenames_test):
        if ii % 10 == 0:
            print(ii+1,'/',len(neither_filenames_test))
        basename = os.path.basename(full_filename)
        dst_filename = os.path.join(test_dir,basename)
        shutil.copyfile(full_filename,dst_filename)
    print("Done")
