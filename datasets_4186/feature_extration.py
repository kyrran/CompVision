import numpy as np
import glob
import cv2
import os
import torch
from skimage.metrics import structural_similarity
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
def get_model():
    model = models.resnet18(pretrained=True)
    #model.classifier = torch.nn.Sequential()
    model.eval()
    return model

def feature_extract(filepath,filename,savepath,model):
    model.eval()

    img = cv2.imread(filepath)
    img = img[:,:,::-1] #bgr2rgb
    img_resize = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)

    trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    #transform to tensor and normalize it
    img_tensor = trans(img_resize)
    #print(data)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    
    feature = model(img_tensor)
    #print(feature)
    feature_numpy = feature.cpu().detach().numpy()
    print(feature_numpy)
    np.save(savepath + filename, feature_numpy)

def main():  
#-------------------------------------crop query----------------------------------------------------------------------------#
    model = get_model()
#----------------------------- to extract feature from cropped query--------------------------------------------------------------------------#
    save_crop_path = r'./query_crop_4186'
    query_feature_savepath = './query_feature_4186/'
    #extract query features and save to folder
    for per_crop_query_file in sorted(glob.glob(save_crop_path+'/*.jpg')):
        #xxx.jpg
        per_crop_query_file_name = os.path.basename(per_crop_query_file)
        #xxx
        per_crop_query_file_name_only = os.path.splitext(per_crop_query_file_name)[0]
        feature_extract(per_crop_query_file,per_crop_query_file_name_only,query_feature_savepath,model)
        
#-----------------------------to extract feature from gallery--------------------------------------------------------------------------#
    path_gallery=r'./gallery_4186'
    gallery_feature_savepath = './gallery_feature_4186/'
    #extract query features and save to folder
    for per_gallery_file in sorted(glob.glob(path_gallery+'/*.jpg')):
        #xxx.jpg
        per_gallery_file_name = os.path.basename(per_gallery_file)
        #xxx
        per_gallery_name_only = os.path.splitext(per_gallery_file_name)[0]
        feature_extract(per_gallery_file,per_gallery_name_only,gallery_feature_savepath,model)

if __name__=='__main__':
    main()
