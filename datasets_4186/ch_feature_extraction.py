import numpy as np
import glob
import cv2
import os
from matplotlib import pyplot as plt

IMAGE_SIZE = 225

LEVEL = 32 #number of bins; number of intervals
INTERVAL = 256 / LEVEL #value range in each interval

def get_features(file):
    image = cv2.imread(file)
    image = image[:,:,::-1] #bgr2rgb
    image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_CUBIC)
    
    numpydata = np.asarray(image)
    #print(numpydata)
    #calculate the frequency of pixels in corresponding intensity
    #let bin = 32, every 8 is an interval, group number 0-31
    
    rows = numpydata.shape[0]
    cols = numpydata.shape[1]
    #1d numpy array
    query_features = np.zeros(LEVEL*LEVEL*LEVEL,int)
    #image size is 225*225 ;loop from 0-224
    for row in range(0,rows):
        for col in range(0,cols,3):
            #print(row,col)
            r_group =int(numpydata[row,col][0]/INTERVAL)
            g_group =int(numpydata[row,col][1]/INTERVAL)
            b_group =int(numpydata[row,col][2]/INTERVAL)
            query_features[(r_group * LEVEL * LEVEL) + (g_group * LEVEL) + b_group] += 1
    return query_features
    
    query_list = query_features.tolist()
    return query_list
    #print(query_features.tolist())


def main():  
#-------------------------------------crop query----------------------------------------------------------------------------#
   
#----------------------------- to extract feature from cropped query--------------------------------------------------------------------------#
    save_crop_path = r'./query_crop_4186'
    path_gallery=r'./gallery_4186'

    #extract query features
    for per_crop_query_file in sorted(glob.glob(save_crop_path+'/*.jpg')):
        file_name = os.path.basename(per_crop_query_file)
        per_crop_query_file_name_only = os.path.splitext(file_name)[0]
        list2_q = get_features(per_crop_query_file)
        np.save('./ch_query_feature_4186/' + per_crop_query_file_name_only,list2_q)

        
    #extract gallery features
    for per_gallery_file in sorted(glob.glob(path_gallery+'/*.jpg')):
        #xxx.jpg
        per_gallery_file_name = os.path.basename(per_gallery_file)
        #xxx
        per_gallery_name_only = os.path.splitext(per_gallery_file_name)[0]
        
        list1_i = get_features(per_gallery_file)
        np.save('./ch_gallery_feature_4186/' + per_gallery_name_only,list1_i)


        
    
                    



    

if __name__=='__main__':
    main()
