import numpy as np
import os
import cv2
import glob
from matplotlib import pyplot as plt


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
    
    counter = 0
    for per_query_feature_file in sorted(glob.glob(r'./ch_query_feature_4186'+'/*.npy')):
        per_query_feature_file_name = os.path.basename(per_query_feature_file)
        per_query_feature_file_name_only = os.path.splitext(per_query_feature_file_name)[0]

        list2_q = np.load(per_query_feature_file)
        #print(per_query_npy)
        
        dictionary = {}
        for per_gallery_feature_file in sorted(glob.glob(r'./ch_gallery_feature_4186'+'/*.npy')):      
            
            list1_i = np.load(per_gallery_feature_file)
            
            #xxx.npy
            per_gallery_feature_file_name = os.path.basename(per_gallery_feature_file)
            #xxx
            per_gallery_feature_name_only = os.path.splitext(per_gallery_feature_file_name)[0]
            
            sim = np.dot(list1_i, list2_q)/(np.linalg.norm(list1_i)*np.linalg.norm(list2_q))
            
            #print(sim)
            dictionary.update({per_gallery_feature_name_only:sim})
        
        counter += 1
        sorted_dict = sorted(dictionary.items(), key=lambda item: item[1]) # Sort the similarity score
        best_ten = sorted_dict[-10:] # Get the best 10 small-big
        best_ten.reverse()#big2small
        print(best_ten)
        
        path = os.path.join('./query_crop_4186/', per_query_feature_file_name_only + '.jpg')
        visulization(best_ten, path) # Visualize the retrieval results
        f = open(r'./rank_list_ch.txt','a')
        f.write('Q'+str(counter)+': ')
        for j in range(10):
            f.write(best_ten[j][0] + ' ')
        f.write('\n')
        f.close()

        



if __name__ == '__main__':
    main()