import subprocess, os, sys
import argparse
import getsignal_lib as lib
import numpy as np
import cv2, time
from urllib.parse import urlparse


displayScreen = False   # Do you want to show the image on LCD?
outputToFile = True     # output the predicted result to image or video file

# Finger amount
TargetFin = 4
# Posture continue time
TimeFin = 3
# Hand area in frame
hand_width = 200
hand_height = 200
# default 0, the frame in center, "-" up and left, "+" down and right
hand_move_h = 0   # horizontal
hand_move_v = 100    # vertical
# Object area in frame
obj_width = 200
obj_height = 100
# default 0, the frame in center, "-" up and left, "+" down and right
obj_move_h = 0      # horizontal
obj_move_v = -75   # vertical

parser = argparse.ArgumentParser(description="Only accept frame get from url video.")
parser.add_argument("-i", dest='image', action='store', help='url video frame')

args = parser.parse_args()

if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_frame.png'
    outputCropFile = args.image[:-4]+'_crop.png'
    outputTargetFile = args.image[:-4]+'_target.png'
    tmptxt = args.image[:-4]+'.txt'

else:
    print("Not the correct input.")
    sys.exit(1)

# get frame border
# Hand rectangle coordinate
hand_up_left_x, hand_up_left_y, hand_btm_right_x, hand_btm_right_y = lib.frame_border(cap, hand_width, hand_height, hand_move_h, hand_move_v)
# Object rectangle coordinate
obj_up_left_x, obj_up_left_y, obj_btm_right_x, obj_btm_right_y = lib.frame_border(cap, obj_width, obj_height, obj_move_h, obj_move_v)

i = 0
ShowHand_flg = False
CntFin = [0, 0, 0, 0, 0, 0] # Count the time of fingers appear

while cv2.waitKey(1) < 0:
    executeCmdflg = False
    hasFrame, frame = cap.read()
    
    if not hasFrame:
        sys.exit(1)

    crop_frame = frame[hand_up_left_y:hand_btm_right_y, hand_up_left_x:hand_btm_right_x] # (y, x)
    object_frame = frame[obj_up_left_y:obj_btm_right_y, obj_up_left_x:obj_btm_right_x] # (y, x)

    crop_frame = lib.grdetect(crop_frame, verbose = True, hand = True, cnt = CntFin)

    cv2.rectangle(frame, (hand_up_left_x, hand_up_left_y), (hand_btm_right_x, hand_btm_right_y), (0, 255, 0), 2)
    cv2.rectangle(frame, (obj_up_left_x, obj_up_left_y), (obj_btm_right_x, obj_btm_right_y), (0, 0, 255), 2)
    
    if (args.image):

        if(outputToFile):  
            lib.DetectObject(tmptxt, outputTargetFile, object_frame, CntFin, TargetFin, TimeFin)
            cv2.imwrite(outputFile, frame.astype(np.uint8))
            cv2.imwrite(outputCropFile, crop_frame.astype(np.uint8))

        if (displayScreen):
            cv2.imshow("frame", frame)
            cv2.imshow("crop_frame", crop_frame)
            cv2.imshow("object_frame", object_frame)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            

