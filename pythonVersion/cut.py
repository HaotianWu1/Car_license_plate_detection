import cv2
import numpy as np
import os
import shutil
from cutLetters import cutLettersWithRefinement

##########################################
#
#  parameter setting
#
##########################################

# image name
img_name = 'TN52U1580.png'

# threshold to perform binary mask
# decrease it if there are area that you don't want to take into consideration
# increase it if parts of letters are lost
# default = 0.55
mask_threshold = 0.3
# ratio for secondary refinement
# decrease it if the picture rotates too much in the secondary refinement 
# increase it if the secondary refinement is not enough 
second_refine_ratio = 0.5

# if perform side cut, cut the top and the bottom of the plate
# which can avoid the negative effect of uncleaned plate side
# set it False if you lost parts of letters
sideCut = True

# if use debug mode
# you can see if your parameter setting is good step by step
# default = True when cut only one number plate
debug = True

##########################################
#
#  parameter setting
#
##########################################

if __name__ == '__main__':

    imgs = os.listdir('im')

    img_path = 'im//' + img_name
    final_letters = cutLettersWithRefinement(img_path, mask_threshold, second_refine_ratio,sideCut, debug)
    print("processing image ", img_name)

    dirname = img_name.replace(".","_")

    if os.path.exists("out/"+dirname):
        shutil.rmtree("out/"+dirname)  
    os.mkdir("out/"+dirname)

    string = 10
    for item in final_letters:
        if item.shape[1]<=0:
            continue
        
        cv2.imwrite("out/"+ dirname +"/"+str(string) + ".png", item * 255)
        string += 1