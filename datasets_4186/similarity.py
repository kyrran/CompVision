from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import cv2
import glob
from matplotlib import pyplot as plt
import torch

def similarity(query_feat, gallery_feat):
    sim = cosine_similarity(query_feat, gallery_feat)
    sim = np.squeeze(sim)
    return sim


def visulization(retrived, query):
    plt.subplot(3, 4, 1)
    plt.title('query')
    query_img = cv2.imread(query)
    img_rgb_rgb = query_img[:,:,::-1]
    plt.imshow(img_rgb_rgb)

    for i in range(10):
        img_path = './gallery_4186/' + retrived[i][0] +'.jpg'
        img = cv2.imread(img_path)
        
        img_rgb = img[:,:,::-1]
        plt.subplot(3, 4, i+2)
        plt.title(retrived[i][1])
        plt.imshow(img_rgb)
    plt.show()
def main():

    i = 0
    for per_query_feature_file in sorted(glob.glob(r'./query_feature_4186'+'/*.npy')):
        per_query_feature_file_name = os.path.basename(per_query_feature_file)
        per_query_feature_file_name_only = os.path.splitext(per_query_feature_file_name)[0]

        per_query_npy = np.load(per_query_feature_file)
        #print(per_query_npy)
        
        dictionary = {}
        for per_gallery_feature_file in sorted(glob.glob(r'./gallery_feature_4186'+'/*.npy')):      
            
            per_gallery_npy = np.load(per_gallery_feature_file)
           
            sim = similarity(per_query_npy, per_gallery_npy)
            
            #xxx.npy
            per_gallery_feature_file_name = os.path.basename(per_gallery_feature_file)
            #xxx
            per_gallery_feature_name_only = os.path.splitext(per_gallery_feature_file_name)[0]
            
            #gallery_path_jpg = os.path.join('./gallery_4186/', per_gallery_feature_name_only+'.jpg')
            #dictionary.update({gallery_path_jpg:sim})
            dictionary.update({per_gallery_feature_name_only:sim})
        #print(dictionary)
        i += 1
        sorted_dict = sorted(dictionary.items(), key=lambda item: item[1]) # Sort the similarity score
        
        best_ten = sorted_dict[-10:] # Get the best 10 small-big
        best_ten.reverse()#big2small
        print(best_ten)
        print("q" + str(i)+ per_query_feature_file + "end")
        path = os.path.join('./query_crop_4186/', per_query_feature_file_name_only+'.jpg')
        visulization(best_ten, path) # Visualize the retrieval results
        f = open(r'./rank_list.txt','a')
        f.write('Q'+str(i)+': ')
        for j in range(10):
            f.write(best_ten[j][0] + ' ')
        f.write('\n')
        f.close()

        



if __name__ == '__main__':
    main()