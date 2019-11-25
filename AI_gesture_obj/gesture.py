import os, sys
import numpy as np
import cv2, time
import pafy
import utils.gesture_util as lib

from config.gesture_config import FLAGS
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


def outputpath(path):
    outputFile = path.split(".")[0] + "_frame." + path.split(".")[-1]
    outputCropFile = path.split(".")[0] + "_crop." + path.split(".")[-1]
    outputTargetFile = path.split(".")[0] + "_target.jpg"
    return outputFile, outputCropFile, outputTargetFile


def gcloudVision(path, lience=FLAGS.glience):
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
    print(item)
    func.save_txt(FLAGS.savetxt_path + "/time.txt", str(item), string="\n" + str(datetime.datetime.now()) + "\n")
                    
    # return label_dict


def main():
    if FLAGS.VideoType == "IMG":
        if not os.path.isfile(FLAGS.infilepath):
            log.info("Input image file " + FLAGS.infilepath + " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(FLAGS.infilepath)
        outputFile, outputCropFile, outputTargetFile = outputpath(FLAGS.infilepath)

    elif FLAGS.VideoType == "VIDEO":
        if not os.path.isfile(FLAGS.infilepath):
            log.info("Input image file " + FLAGS.infilepath + " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(FLAGS.infilepath)
        outputFile, outputCropFile, outputTargetFile = outputpath(FLAGS.infilepath)
        if FLAGS.outputToFile:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            outcrop = cv2.VideoWriter(outputCropFile, fourcc, 30.0, (round(FLAGS.hand_width),round(FLAGS.hand_height)))
    
    elif FLAGS.VideoType == "URL":
        ret = urllib.request.urlopen(FLAGS.url_source)
        if ret.code != 200:
            log.info("URL not exist: ", FLAGS.url_source)
            sys.exit(1)
        vPafy = pafy.new(FLAGS.url_source)
        play = vPafy.getbest()
        video_source = play.url
        im_width, im_height = vc.get_video_size(video_source)
        process1 = vc.start_ffmpeg_process1(video_source) 
        outputFile, outputCropFile, outputTargetFile = outputpath(FLAGS.url_outfilepath)
        if FLAGS.outputToFile:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(im_width),
                round(im_height)))
            outcrop = cv2.VideoWriter(outputCropFile, fourcc, 30.0, (round(FLAGS.hand_width),round(FLAGS.hand_height)))
    
    elif FLAGS.VideoType == "WEBCAM":
        cap = cv2.VideoCapture(0)
        outputFile, outputCropFile, outputTargetFile = outputpath(FLAGS.url_outfilepath)
        if FLAGS.outputToFile:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(outputFile, fourcc, 30.0, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            outcrop = cv2.VideoWriter(outputCropFile, fourcc, 30.0, (round(FLAGS.hand_width),round(FLAGS.hand_height)))

    else:
        log.info(" error : TYPE is {} ".format(FLAGS.VideoType))
        sys.exit(1)


    i = 0
    ShowHand_flg = False
    CntFin = [0, 0, 0, 0, 0, 0] # Count the time of fingers appear

    start = time.time()
    while cv2.waitKey(1) < 0:
        if FLAGS.VideoType == "URL":
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

        if FLAGS.VideoType == "IMG":
            frame = cv2.resize(frame, (FLAGS.cap_width, FLAGS.cap_height))

        # Hand location
        hand_L, hand_T, hand_R, hand_B = \
            lib.frame_border(frame, FLAGS.hand_width, FLAGS.hand_height, FLAGS.hand_move_h, FLAGS.hand_move_v)
        # Object location
        obj_L, obj_T, obj_R, obj_B = \
            lib.frame_border(frame, FLAGS.obj_width, FLAGS.obj_height, FLAGS.obj_move_h, FLAGS.obj_move_v)

        
        i += 1
        hand_frame = frame[hand_T:hand_B, hand_L:hand_R] # (y, x)
        obj_frame = frame[obj_T:obj_B, obj_L:obj_R] # (y, x)

        if FLAGS.detecthand:
            if FLAGS.VideoType == "IMG":
                hand_frame = lib.grdetect(hand_frame, verbose = True, hand = True, cnt = CntFin)
            elif FLAGS.VideoType in ["VIDEO", "URL", "WEBCAM"]:
                # control time of detection, counting 36 frames almost 1 second
                if i%36 == 0:
                    hand_frame = lib.grdetect(hand_frame, verbose = True, hand = True, cnt = CntFin)
                    log.info("Pass {:.3f} seconds".format(time.time() - start))
                else:
                    hand_frame = lib.grdetect(hand_frame, verbose = True, hand = False, cnt = CntFin)
        

        if FLAGS.VideoType == "IMG":
            if FLAGS.detectobject:
                if (CntFin[FLAGS.TargetFin] >= 1):
                    cv2.imwrite(outputTargetFile, obj_frame.astype(np.uint8))
                    labels = gcloudVision(path=outputTargetFile, lience=FLAGS.glience)
                    CntFin = [0, 0, 0, 0, 0, 0]
                    # label: object, score, vertice
                    item = []
                    [item.append(labels["object"][labels["score"].index(i)]) for i in labels["score"] if i > 0.8]
                    print(item)

            if FLAGS.outputToFile:   
                cv2.imwrite(outputCropFile, hand_frame.astype(np.uint8))
                # Draw hand frame on image
                cv2.rectangle(frame, (hand_L, hand_T), (hand_R, hand_B), (0, 255, 0), 2)
                # Draw object frame on image
                cv2.rectangle(frame, (obj_L, obj_T), (obj_R, obj_B), (0, 0, 255), 2)
                cv2.imwrite(outputFile, frame.astype(np.uint8))

            if FLAGS.displayScreen:
                cv2.imshow("frame", frame)
                cv2.imshow("hand_frame", hand_frame)
                cv2.imshow("object_frame", obj_frame)
                cv2.waitKey(10000)
                cv2.destroyAllWindows()

        elif FLAGS.VideoType in ["VIDEO", "URL", "WEBCAM"]:
            if FLAGS.detectobject:
                if (CntFin[FLAGS.TargetFin] >= FLAGS.TimeFin):
                    cv2.imwrite(outputTargetFile, obj_frame.astype(np.uint8))
                    log.info("Start detect object.")
                    threading.Thread(target=gcloudVision, args=(outputTargetFile, FLAGS.glience)).start()
                    CntFin = [0, 0, 0, 0, 0, 0]

            # Draw hand frame on image
            cv2.rectangle(frame, (hand_L, hand_T), (hand_R, hand_B), (0, 255, 0), 2)
            # Draw object frame on image
            cv2.rectangle(frame, (obj_L, obj_T), (obj_R, obj_B), (0, 0, 255), 2)
                
            if FLAGS.outputToFile:
                outcrop.write(hand_frame)
                out.write(frame)
            
            if FLAGS.displayScreen:
                cv2.imshow("frame", frame)
                cv2.imshow("hand_frame", hand_frame)
                cv2.imshow("object_frame", obj_frame)
                cv2.waitKey(15)
        

if __name__ == '__main__':
    main()
