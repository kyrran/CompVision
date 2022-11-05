from signal import Sigmasks
from sre_constants import IN_LOC_IGNORE
import psnr_cal
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def drawlines(imgL,imgR,lines,pts1,pts2):
    r,c = imgL.shape
    imgL = cv.cvtColor(imgL,cv.COLOR_GRAY2BGR)
    imgR = cv.cvtColor(imgR,cv.COLOR_GRAY2BGR)
    np.random.seed(0)

    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())

        #x0,y0 = map(int, [0, -r[2]/r[1] ])
        #x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

        #img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        imgL = cv.circle(imgL,tuple(pt1),5,color,-1)
        imgR = cv.circle(imgR,tuple(pt2),5,color,-1)

        # print("p1:" +str(imgL[pt1[0]][pt1[1]]))
        # print("p2:" + str(imgR[pt2[0]][pt2[1]]))
        # print("dis:" + str(imgL[pt1[0]][pt1[1]] - imgR[pt2[0]][pt2[1]]))
        
    return imgL,imgR


def main():

    test_imgs = ["Art", "Dolls", "Reindeer"]    
    
    for index in range(3):
        ln = "./"+test_imgs[index]+"/view1.png"
        imgL = cv.imread(ln,0)
       
        rn = "./"+test_imgs[index]+"/view5.png"
        imgR = cv.imread(rn,0)

        dn = "./"+test_imgs[index]+"/disp2.png"
        gtn = "./"+test_imgs[index]+"/disp1.png"

        unl = "./"+test_imgs[index]+"/undistorted_L.png"
        unr = "./"+test_imgs[index]+"/undistorted_R.png"


        sift = cv.SIFT_create()
        # keyPoint1 = []
        # keyPoint2 = []
        # for i in range (imgL.shape[0]):
        #     for j in range (imgL.shape[1]):
        #         keyPoint1.append(cv.KeyPoint(i,j, 1))
        #         keyPoint2.append(cv.KeyPoint(i,j, 1))
        
        # keyPoint1, descriptor1 = sift.compute(imgL,keyPoint1)
        # keyPoint2, descriptor2 = sift.compute(imgR,keyPoint2)

        keyPoint1, descriptor1 = sift.detectAndCompute(imgL,None)
        keyPoint2, descriptor2 = sift.detectAndCompute(imgR,None)

        # kparray = cv.drawKeypoints(imgL, keyPoint1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow("Image", kparray)

        # kparray = cv.drawKeypoints(imgR, keyPoint2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imshow("Image", kparray)

        

        # BFMatcher with default params
        # bf = cv.BFMatcher()
        # matches = bf.knnMatch(descriptor1,descriptor2,k=2)
        # Apply ratio test
        # good = []
        # for m,n in matches:
        #     if m.distance < 0.75*n.distance:
        #         good.append([m])


            
        #FLANN Parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descriptor1,descriptor2,k=2)

        psL = []
        psR = []
        # good =[]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                # good.append([m])
                psR.append(keyPoint2[m.trainIdx].pt)
                psL.append(keyPoint1[m.queryIdx].pt)

        psL = np.int32(psL)
        psR = np.int32(psR)

        
        fundamental_matrix, inliers = cv.findFundamentalMat(psL,psR,cv.FM_LMEDS)

        #inliers
        psL = psL[inliers.ravel()==1]
        psR = psR[inliers.ravel()==1]

        # epilines to points in right
        # draw on left image
        lines1 = cv.computeCorrespondEpilines(psR.reshape(-1,1,2), 2,fundamental_matrix)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = drawlines(imgL,imgR,lines1,psL,psR)


        # epilines to points in left
        # draw on right image
        # lines2 = cv.computeCorrespondEpilines(psL.reshape(-1,1,2), 1,fundamental_matrix)
        # lines2 = lines2.reshape(-1,3)
        # img3,img4 = drawlines(imgR,imgL,lines2,psR,psL)
        h1, w1 = imgL.shape
        h2, w2 = imgR.shape
        thresh = 0
        _, H1, H2 = cv.stereoRectifyUncalibrated(
            np.float32(psL), np.float32(psR), fundamental_matrix, imgSize=(w1, h1), threshold=thresh,
        )
        imgL_undistorted = cv.warpPerspective(imgL, H1, (w1, h1))
        imgR_undistorted = cv.warpPerspective(imgR, H2, (w2, h2))
        
        cv.imwrite(unl, imgL_undistorted)
        cv.imwrite(unr, imgR_undistorted)

        
        # plt.subplot(121),plt.imshow(img5)
        # plt.subplot(122),plt.imshow(img6)
        # plt.show()


        stereo = cv.StereoSGBM_create(
            minDisparity=7,
            numDisparities=3,
            blockSize=2,
            uniquenessRatio=0,
            speckleWindowSize=0,
            speckleRange=20,
            P1=3*8*2**2,
            P2=3*32*2**2,
        )   
        #stereo = cv.StereoBM_create(numDisparities=16, blockSize=5)
        disparity = stereo.compute(imgL_undistorted,imgR_undistorted)
        cv.imwrite(dn,disparity)
        psnr_cal.test(gtn,dn)
        plt.imshow(disparity,'gray')
        plt.show()
        
        # rightmathcer = cv.ximgproc.createRightMatcher(stereo)

        # lmbda = 80000
        # sigma = 1.2
        # visual_multi = 1.0

        # wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left = stereo)
        # wls_filter.setLambda(lmbda)
        # wls_filter.setSigmaColor(sigma)

        # displ = stereo.compute(imgL_undistorted,imgR_undistorted)
        # dispr = rightmathcer.compute(imgR_undistorted,imgL_undistorted)

        # displ = np.int16(displ)
        # dispr = np.int16(dispr)
        # filterediMG = wls_filter.filter(displ,imgL_undistorted,None,dispr)
        # filterediMG = cv.normalize(src = filterediMG, dst = filterediMG, beta = 0, alpha = 255)
        # filterediMG = np.array(filterediMG)
        # cv.imshow('hi',filterediMG)
        # plt.show()

        # cv.imwrite(dn,filterediMG)
        # psnr_cal.test(gtn,dn)
        

    
if __name__== '__main__':
    main()