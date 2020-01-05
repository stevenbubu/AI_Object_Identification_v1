import os, sys
import numpy as np
import cv2, time
import pafy
import utils.gesture_util as lib

import logging
import config.log_config as logcfg
logging.config.dictConfig(logcfg.config)
log = logging.getLogger("StreamLogger")
import utils.video_capture as vc
import urllib.request
from queue import Queue
import threading
import datetime
import utils.func_util as func
import utils.config as cfg
import tensorflow as tf

""" FLAGS """
flags = tf.compat.v1.flags
flags.DEFINE_string(name="INIT_CONFIGS_PATH", default="config/gesture.config", help="Config file path.")
FLAGS = flags.FLAGS

def outputpath(path):
    outputFile = path.split(".")[0] + "_frame." + path.split(".")[-1]
    outputCropFile = path.split(".")[0] + "_crop." + path.split(".")[-1]
    outputTargetFile = path.split(".")[0] + "_target.jpg"
    return outputFile, outputCropFile, outputTargetFile


def gcloudVision(path, lience, recordpath):
    import os, io
    from google.cloud import vision

    # Instantiates a client
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = lience

    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    Object = []; Score = []; Vertices = []
    for label in labels:
        Object.append(label.description)
        Score.append(label.score)

    # # Performs text detection on the image file_name
    response = client.text_detection(image=image)
    texts = response.text_annotations
    for text in texts:
        Vertices.append(['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

    label_dict = {}
    label_dict["object"] = Object
    label_dict["score"] = Score
    label_dict["vertice"] = Vertices
    
    item = []
    [item.append(label_dict["object"][label_dict["score"].index(i)]) for i in label_dict["score"] if i > 0.8]
    # print(item)
    func.save_txt(recordpath, str(item), string="\n" + str(datetime.datetime.now()) + "\n")
                    
    # return label_dict

def Hand_Target_Area(option):
    Hand_Area = option["Hand_Area"]
    Hand_Width, Hand_Height = Hand_Area["Width"], Hand_Area["Height"]
    Hand_Shift = Hand_Area["Shift"]
    Hand_move_Horizontial, Hand_move_Vertical = Hand_Shift["Horizontial"], Hand_Shift["Vertical"]

    Object_Area = option["Object_Area"]
    Object_Width, Object_Height = Object_Area["Width"], Object_Area["Height"]
    Object_Shift = Object_Area["Shift"]
    Object_move_Horizontial, Object_move_Vertical = Object_Shift["Horizontial"], Object_Shift["Vertical"]

    return Hand_Width, Hand_Height, Hand_move_Horizontial, Hand_move_Vertical, \
            Object_Width, Object_Height, Object_move_Horizontial, Object_move_Vertical



def main():
    # import config
    InitConfig = cfg.get_config_from_init_config(init_configs_path=FLAGS.INIT_CONFIGS_PATH)
    Flags = InitConfig["Flags"]
    MediaType = Flags["MediaType"]
    GLience = Flags["GLience"]
    # Control flag
    DetectHnad = Flags["DetectHnad"]
    DetectObject = Flags["DetectObject"]
    OutputToFile = Flags["OutputToFile"]
    DisplayScreen = Flags["DisplayScreen"]

    start = time.time()
    # Initial Type setting
    if MediaType == "IMAGE":
        Config = Flags['IMAGE_CONFIG']
        TextPath = Config["TextPath"]
        if all(isinstance(x,float) for x in Hand_Target_Area(Config)):
            Hand_Width, Hand_Height, Hand_move_Horizontial, Hand_move_Vertical, \
            Object_Width, Object_Height, Object_move_Horizontial, Object_move_Vertical \
            = Hand_Target_Area(Config)
        else:
            log.info("MediaType: " + MediaType + ", Hand or Object area setting error.")
            sys.exit(1)

        indir = os.path.join(Config["InputPath"], Config["InputFile"])
        if os.path.isfile(indir):
            cap = cv2.VideoCapture(indir)
            outputFile, outputCropFile, outputTargetFile = outputpath(indir)
        else:
            log.info("MediaType: " + MediaType + ", indir: " + indir + " error.")
            sys.exit(1)

    elif MediaType == "VIDEO":
        Config = Flags['VIDEO_CONFIG']
        TextPath = Config["TextPath"]
        if all(isinstance(x,float) for x in Hand_Target_Area(Config)):
            Hand_Width, Hand_Height, Hand_move_Horizontial, Hand_move_Vertical, \
            Object_Width, Object_Height, Object_move_Horizontial, Object_move_Vertical \
            = Hand_Target_Area(Config)
        else:
            log.info("MediaType: " + MediaType + ", Hand or Object area setting error.")
            sys.exit(1)

        indir = os.path.join(Config["InputPath"], Config["InputFile"])
        if os.path.isfile(indir):
            cap = cv2.VideoCapture(indir)
            outputFile, outputCropFile, outputTargetFile = outputpath(indir)
            if OutputToFile:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                    round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                outcrop = cv2.VideoWriter(outputCropFile, fourcc, 30.0, (round(Hand_Width),round(Hand_Height)))
        else:
            log.info("MediaType: " + MediaType + ", indir: " + indir + " error.")
            sys.exit(1)

    elif MediaType == "URL":
        Config = Flags['URL_CONFIG']
        TextPath = Config["TextPath"]
        if all(isinstance(x,float) for x in Hand_Target_Area(Config)):
            Hand_Width, Hand_Height, Hand_move_Horizontial, Hand_move_Vertical, \
            Object_Width, Object_Height, Object_move_Horizontial, Object_move_Vertical \
            = Hand_Target_Area(Config)
        else:
            log.info("MediaType: " + MediaType + ", Hand or Object area setting error.")
            sys.exit(1)

        URL = Config["URL"]
        outdir = os.path.join(Config["OutputPath"], Config["OutputFile"])
        if urllib.request.urlopen(URL).code == 200:
            vPafy = pafy.new(URL)
            play = vPafy.getbest()
            video_source = play.url
            im_width, im_height = vc.get_video_size(video_source)
            process1 = vc.start_ffmpeg_process1(video_source) 
            outputFile, outputCropFile, outputTargetFile = outputpath(outdir)
            if OutputToFile:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                # out = cv2.VideoWriter("/home/jdwei/Desktop/github/AI_Object_Identification_v1/AI_gesture_obj_id/images/url_output.mp4", fourcc, 30.0, (round(im_width), round(im_height)))
                out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(im_width), round(im_height)))
                outcrop = cv2.VideoWriter(outputCropFile, fourcc, 30.0, (round(Hand_Width), round(Hand_Height)))
        else:
            log.info("MediaType: " + MediaType + ", URL: " + URL + " not exist.")
            sys.exit(1)

    elif MediaType == "WEBCAM":
        Config = Flags['WEBCAM_CONFIG']
        TextPath = Config["TextPath"]
        if all(isinstance(x,float) for x in Hand_Target_Area(Config)):
            Hand_Width, Hand_Height, Hand_move_Horizontial, Hand_move_Vertical, \
            Object_Width, Object_Height, Object_move_Horizontial, Object_move_Vertical \
            = Hand_Target_Area(Config)
        else:
            log.info("MediaType: " + MediaType + ", Hand or Object area setting error.")
            sys.exit(1)

        WEBCAM = int(Config["WEBCAM"])
        outdir = os.path.join(Config["OutputPath"], Config["OutputFile"])
        if isinstance(WEBCAM,(int)):
            cap = cv2.VideoCapture(WEBCAM)
            outputFile, outputCropFile, outputTargetFile = outputpath(outdir)
            if OutputToFile:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                outcrop = cv2.VideoWriter(outputCropFile, fourcc, 30.0, (round(Hand_Width),round(Hand_Height)))
        else:
            log.info("MediaType: " + MediaType + ", WEBCAM: " + WEBCAM + " error.")
            sys.exit(1)

    else:
        log.info("MediaType: " + MediaType + " error.")
        sys.exit(1)

  
    # Process media
    i = 0
    ShowHand_flg = False
    CntFin = [0, 0, 0, 0, 0, 0] # Count the time of fingers appear

    roundtime = time.time()
    while cv2.waitKey(1) < 0:
        if MediaType == "URL":
            image_np = vc.read_frame(process1, im_width, im_height)
            # use ffmpeg need to complement color
            image_np_tmp = np.zeros(image_np.shape, dtype=np.uint8)
            # m  b
            m_r, b_r = 0.9900855673711094, 2.05757036126613
            m_g, b_g = 0.9880223010384511, 1.9458285218824187
            m_b, b_b = 0.9882212111398636, 1.777887000339831
            image_np_tmp[:,:,0] = image_np[:,:,0]*m_r + b_r
            image_np_tmp[:,:,1] = image_np[:,:,1]*m_g + b_g
            image_np_tmp[:,:,2] = image_np[:,:,2]*m_b + b_b
            # BGR
            frame = np.dstack((image_np_tmp[:,:,2],image_np_tmp[:,:,1], image_np_tmp[:,:,0]))

        else:
            hasFrame, frame = cap.read()
            if not hasFrame:
                log.info("Done processing !" + \
                        "\nOutput file is stored as " + outputFile + \
                        "\nOutput crop file is stored as " + outputCropFile + \
                        "\nOutput target file is stored as " + outputTargetFile)
                cv2.waitKey(1000)
                sys.exit(1)

        if MediaType == "IMAGE":
            Resize = Config["Resize"]
            Resize_Width, Resize_Height = int(Resize["Width"]), int(Resize["Height"])
            frame = cv2.resize(frame, (Resize_Width, Resize_Height))


        # Hand location
        hand_L, hand_T, hand_R, hand_B = \
            lib.frame_border(frame, Hand_Width, Hand_Height, Hand_move_Horizontial, Hand_move_Vertical)
        # Object location
        obj_L, obj_T, obj_R, obj_B = \
            lib.frame_border(frame, Object_Width, Object_Height, Object_move_Horizontial, Object_move_Vertical)

        i += 1
        hand_frame = frame[hand_T:hand_B, hand_L:hand_R]    # (y, x)
        obj_frame = frame[obj_T:obj_B, obj_L:obj_R]         # (y, x)


        if MediaType == "IMAGE":
            Finger = Config["Finger"]
            Target = int(Finger["Target"])
            if DetectHnad:
                hand_frame = lib.grdetect(hand_frame, verbose = True, hand = True, cnt = CntFin)

            if DetectObject:
                if (CntFin[Target] >= 1):
                    cv2.imwrite(outputTargetFile, obj_frame.astype(np.uint8))
                    gcloudVision(path=outputTargetFile, lience=GLience, recordpath=TextPath)
                    CntFin = [0, 0, 0, 0, 0, 0]
            
            # Draw hand frame on image
            cv2.rectangle(frame, (hand_L, hand_T), (hand_R, hand_B), (0, 255, 0), 2)
            # Draw object frame on image
            cv2.rectangle(frame, (obj_L, obj_T), (obj_R, obj_B), (0, 0, 255), 2)

            if OutputToFile:   
                cv2.imwrite(outputCropFile, hand_frame.astype(np.uint8))
                cv2.imwrite(outputFile, frame.astype(np.uint8))

            if DisplayScreen:
                cv2.imshow("frame", frame)
                cv2.imshow("hand_frame", hand_frame)
                cv2.imshow("object_frame", obj_frame)
                cv2.waitKey(10000)
                cv2.destroyAllWindows()

        else:
            Finger = Config["Finger"]
            Target = int(Finger["Target"])
            Time = int(Finger["Time"])
            if DetectHnad:
                # control time of detection, counting 36 frames almost 1 second
                if i%36 == 0:
                    hand_frame = lib.grdetect(hand_frame, verbose = True, hand = True, cnt = CntFin)
                    log.info("Pass {:.3f} seconds".format(time.time() - roundtime))
                else:
                    hand_frame = lib.grdetect(hand_frame, verbose = True, hand = False, cnt = CntFin)

            if DetectObject:
                if (CntFin[Target] >= Time):
                    cv2.imwrite(outputTargetFile, obj_frame.astype(np.uint8))
                    log.info("Start detect object.")
                    threading.Thread(target=gcloudVision, args=(outputTargetFile, GLience, TextPath)).start()
                    CntFin = [0, 0, 0, 0, 0, 0]

            # Draw hand frame on image
            cv2.rectangle(frame, (hand_L, hand_T), (hand_R, hand_B), (0, 255, 0), 2)
            # Draw object frame on image
            cv2.rectangle(frame, (obj_L, obj_T), (obj_R, obj_B), (0, 0, 255), 2)
                
            if OutputToFile:
                outcrop.write(hand_frame)
                out.write(frame)
            
            if DisplayScreen:
                cv2.imshow("frame", frame)
                cv2.imshow("hand_frame", hand_frame)
                cv2.imshow("object_frame", obj_frame)
                cv2.waitKey(15)


if __name__ == '__main__':
    main()
