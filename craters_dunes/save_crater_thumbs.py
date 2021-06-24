import os,glob
import concurrent.futures
from PIL import Image
import pandas as pd
import numpy as np

input_dir = '/home/ahoag/ML/mars/gallery_app/app/static/images'
trainval_dir = '/home/ahoag/ML/mars/craters_dunes/images_for_cnn/craters'
test_dir = '/home/ahoag/ML/mars/craters_dunes/test_cnn/craters'
csv_filename = '/home/ahoag/ML/mars/gallery_app/app/static/multilabel_training_set.csv'

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
    mars_df = pd.read_csv(csv_filename)
    crater_df = mars_df[mars_df.loc[:,'has_crater']==1]
    crater_filenames = crater_df.filename.values
    np.random.seed(42)
    np.random.shuffle(crater_filenames)
    crater_filenames_train_val = crater_filenames[0:750]
    crater_filenames_test = crater_filenames[750:]
    """ Train/val set first """
    for ii,filename in enumerate(crater_filenames_train_val):
        full_filename = os.path.join(input_dir,filename)        
        if ii % 10 == 0:
            print(ii+1,'/',len(crater_filenames_train_val))
        make_thumb(full_filename,trainval_dir)
    """ Now test set """
    for ii,filename in enumerate(crater_filenames_test):
        full_filename = os.path.join(input_dir,filename)        
        if ii % 10 == 0:
            print(ii+1,'/',len(crater_filenames_test))
        make_thumb(full_filename,test_dir)
    print("Done")
