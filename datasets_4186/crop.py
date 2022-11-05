import numpy as np
import glob
import cv2
import os


def crop_query(path_query):

    for per_query_file in sorted(glob.glob(path_query+'/*.jpg')):
        #xxx.jpg
        per_query_file_name = os.path.basename(per_query_file)
        #xxx
        query_name_only = os.path.splitext(per_query_file_name)[0]
        #find all "xxx.txt"
        #the output -> ['query_txt_4186/1258.txt'].......
        per_txt_filepath = glob.glob('**/'+query_name_only+'.txt', recursive=True)
        
        query_img = cv2.imread(per_query_file)
        query_img = query_img[:,:,::-1] #bgr2rgb
        #['query_txt_4186/1258.txt'][0] is the path, not string
        #output -> query_txt_4186/1258.txt
        txt = np.loadtxt(per_txt_filepath[0])#load the coordinates of the bounding box

        crop = query_img[int(txt[1]):int(txt[1] + txt[3]), int(txt[0]):int(txt[0] + txt[2]), :] #crop the instance region
        #save as "xxx.jpg" into croped folder
        crop_path = os.path.join('./query_crop_4186/', per_query_file_name)
        cv2.imwrite(crop_path, crop[:,:,::-1])  #save the cropped region

def main():  
#-------------------------------------crop query----------------------------------------------------------------------------#
    path_query=r'./query_4186'
    crop_query(path_query)

if __name__=='__main__':
    main()
