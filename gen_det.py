import numpy as np
import cv2 as cv
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating masks based on ultrasound scans')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the generated masks folder")

    args = parser.parse_args()
    img_files = sorted(glob(os.path.join(args.img_path, "*jpg")))

    try:

        kernel = np.ones((11, 11), np.uint8)
        for fimg in img_files:
            img = cv.imread(fimg, 0)
            #adding an extra border on the image will help us later as explained in the assignment report
            img2 = cv.copyMakeBorder(img, 100, 100, 100, 100, cv.BORDER_CONSTANT, value=0)

            #performing CLAHE histogram equalization
            #create a CLAHE object (Arguments are optional)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img2 = clahe.apply(img2)
            
            #inverse threshold so that our main gallbladder feature and background gets value 255, rest are 0
            ret, thresh = cv.threshold(img2, 54, 255, cv.THRESH_BINARY_INV) 
            
            #Erosion followed by dilation
            opening = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
            
            #finding all the connected components
            ret, markers = cv.connectedComponents(opening)
            
            #finding the area covered by each connected component
            marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
            
            #largest connected component is the background, set that to 0
            largest_component = np.argmax(marker_area)
            largest_component
            marker_area[largest_component] = 0

            #now the largest connected component is the gall bladder feature
            largest_component = np.argmax(marker_area)
            largest_component = np.argmax(marker_area)+1
            mask = markers==largest_component
            
            #setting all other features to be 0
            out = img2.copy()
            out[mask == False] = 0
            out[mask == True] = 255
            
            #thresholding to find the feature
            ret, thresh2 = cv.threshold(out,0, 255, cv.THRESH_OTSU+cv.THRESH_BINARY)

            #since the mask had some gaps in it, performing a closing
            closing = cv.morphologyEx(thresh2, cv.MORPH_CLOSE, kernel, iterations = 3)

            #removing the borders we made initially
            closing = closing[100:len(closing)-100, 100:len(closing[0])-100]
            # plt.imshow(closing, cmap='gray')
            # plt.show()

            cv.imwrite(os.path.join(args.det_path, os.path.split(fimg)[1]), closing)
    except:
        print("except clause: encountered error")