import subprocess, os, sys
import argparse
import getsignal_lib as lib
import numpy as np
import cv2, time
import pafy, vlc
from urllib.parse import urlparse


displayScreen = True   # Do you want to show the image on LCD?
outputToFile = True    # output the predicted result to image or video file

# Finger amount
TargetFin = 4
# Posture continue time
TimeFin = 2
# Hand area in frame
hand_width = 450
hand_height = 500
# default 0, the frame in center, "-" up and left, "+" down and right
hand_move_h = 0     # horizontal
hand_move_v = 200   # vertical
# Object area in frame
obj_width = 500
obj_height = 300
# default 0, the frame in center, "-" up and left, "+" down and right
obj_move_h = 0      # horizontal
obj_move_v = -200   # vertical

parser = argparse.ArgumentParser(description="Which mode do you want to scan, image, video or live stream?")
parser.add_argument("-i", dest='image', action='store', help='Image')
parser.add_argument("-v", dest='video', action='store', help='Video file')
parser.add_argument("-s", dest='url', action='store', help='utl video stream')

args = parser.parse_args()
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_frame.jpg'
    outputCropFile = args.image[:-4]+'_crop.jpg'
    outputTargetFile = args.image[:-4]+'_target.jpg'

elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_frame.avi'
    outputCropFile = args.video[:-4]+'_crop.avi'
    outputTargetFile = args.video[:-4]+'_target.jpg'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    outcrop = cv2.VideoWriter(outputCropFile, fourcc, 30.0, (round(hand_width),round(hand_height)))

elif (args.url):
    scheme, netloc, path, params, query, fragment = urlparse(args.url)
    if "http" not in scheme:
        print("Input url ", args.url, " doesn't exist")
        sys.exit(1)
        
    executeCmd = "python3 " + sys.path[0] + "/urlvideo.py" + " -s " +  args.url
    lib.call_pyfile(executeCmd)

else:
    print("Not the correct input.")
    sys.exit(1)


if not (args.url):
    # Run YOLOv3 detect object dir
    predir = os.path.abspath(os.path.join(sys.path[0],os.path.pardir)) # parent dir
    Targetimgdir = os.path.abspath(outputTargetFile)

    # get frame border
    # Hand rectangle coordinate
    hand_up_left_x, hand_up_left_y, hand_btm_right_x, hand_btm_right_y = lib.frame_border(cap, hand_width, hand_height, hand_move_h, hand_move_v)
    # Object rectangle coordinate
    obj_up_left_x, obj_up_left_y, obj_btm_right_x, obj_btm_right_y = lib.frame_border(cap, obj_width, obj_height, obj_move_h, obj_move_v)

    i = 0
    ShowHand_flg = False
    CntFin = [0, 0, 0, 0, 0, 0] # Count the time of fingers appear

    start = time.time()
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()

        if not hasFrame:
            print("Done processing !")
            print("Output file is stored as ", outputFile)
            print("Output crop file is stored as ", outputCropFile)
            print("Output target file is stored as ", outputTargetFile)
            cv2.waitKey(3000)
            sys.exit(1)

        i = i + 1
        # control time of detection, counting 36 frames almost 1 second
        if i%36 == 0:
            ShowHand_flg = True
            print("Pass {:.3f} seconds".format(time.time() - start))
        else:
            ShowHand_flg = False

        crop_frame = frame[hand_up_left_y:hand_btm_right_y, hand_up_left_x:hand_btm_right_x] # (y, x)
        object_frame = frame[obj_up_left_y:obj_btm_right_y, obj_up_left_x:obj_btm_right_x] # (y, x)

        if (args.image):
            crop_frame = lib.grdetect(crop_frame, verbose = True, hand = True, cnt = CntFin)
        elif (args.video):
            crop_frame = lib.grdetect(crop_frame, verbose = True, hand = ShowHand_flg, cnt = CntFin)

        cv2.rectangle(frame, (hand_up_left_x, hand_up_left_y), (hand_btm_right_x, hand_btm_right_y), (0, 255, 0), 2)
        cv2.rectangle(frame, (obj_up_left_x, obj_up_left_y), (obj_btm_right_x, obj_btm_right_y), (0, 0, 255), 2)
        
        if (args.image):

            if(outputToFile):     
                if (CntFin[TargetFin] >= 1):
                    cv2.imwrite(outputTargetFile, object_frame.astype(np.uint8))
                    for p in range(len(CntFin)):
                        CntFin[p] = 0  
                    # Run YOLOv3 detect object
                    executeCmd = "python3 " + predir + "/yolov3_608/playYOLO.py" + " -i " +  Targetimgdir
                    lib.call_pyfile(executeCmd)

                cv2.imwrite(outputFile, frame.astype(np.uint8))
                cv2.imwrite(outputCropFile, crop_frame.astype(np.uint8))

            if (displayScreen):
                cv2.imshow("frame", frame)
                cv2.imshow("crop_frame", crop_frame)
                cv2.imshow("object_frame", object_frame)
                cv2.waitKey(10000)
                cv2.destroyAllWindows()

        else:

            if(outputToFile):
                if (CntFin[TargetFin] >= TimeFin):
                    cv2.imwrite(outputTargetFile, object_frame.astype(np.uint8))
                    for p in range(len(CntFin)):
                        CntFin[p] = 0
                    # Run YOLOv3 detect object
                    executeCmd = "python3 " + predir + "/yolov3_608/playYOLO.py" + " -i " +  Targetimgdir
                    lib.call_pyfile(executeCmd)
                
                out.write(frame)
                outcrop.write(crop_frame)
            
            if(displayScreen):
                cv2.imshow("frame", frame)
                cv2.imshow("crop_frame", crop_frame)
                cv2.imshow("object_frame", object_frame)
                cv2.waitKey(1)