import cv2
import numpy as np
import os
import shutil
from cutLetters import readCutLetters, readtemplates, matching
import cut

##########################################
#
#  parameter setting
#
##########################################

# image name, already set in the cut.py
img_name = cut.img_name

# threshold for making cut letteres binary
# default is same as mask_threshold in cut.py, but you can change it
# decrease it if there are areas that you don't want to take into consideration
# increase it if parts of letters are lost
binary_letters_threshold =  cut.mask_threshold

# down sample scale
# increase to get faster performance
# decrease to get better result
# default = 4, must be an integer
down_sample_scale = 4

# penalty for one pixel if letter doesn't match the template
# default = 0.2
penalty = 0.2

##########################################
#
#  parameter setting
#
##########################################


if __name__ == '__main__':



    img_path = 'out//' + img_name.replace(".","_")

    final_binary_letters = readCutLetters(img_path, down_sample_scale, binary_letters_threshold)
    template_letters, templates = readtemplates(down_sample_scale, penalty)
    results = matching(template_letters, templates, final_binary_letters)

    print("==========\nmatching result for each letter:")
    print(results)
    print("==========\nfinal result:")
    print("".join(results))