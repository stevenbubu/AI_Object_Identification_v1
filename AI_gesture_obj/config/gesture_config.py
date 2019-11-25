import os 

class FLAGS(object):
    
    # General parameter
    VideoType = "WEBCAM"      # IMG, VIDEO, URL, WEBCAM
    infile = "hand_video.mp4"
    inpath = "/home/jdwei/Desktop/github/AI_Object_Identification_v1/AI_gesture_obj_id/images"
    infilepath = os.path.join(inpath, infile)

    url_outfile = "url_output.mp4"
    url_outpath = "/home/jdwei/Desktop/github/AI_Object_Identification_v1/AI_gesture_obj_id/images"
    url_outfilepath = os.path.join(inpath, infile)
    url_source = "https://www.youtube.com/watch?v=8tpxwFEIxi0"

    savetxt_path = "/home/jdwei/Desktop/github/AI_Object_Identification_v1/AI_gesture_obj_id/images"
    # resize input image
    cap_width = 640
    cap_height = 480

    # Google API Lience
    glience = "/home/jdwei/Desktop/github/AI_Object_Identification_v1/AI_gesture_obj_id/Gesture-Object-Recognition-a90ad9774b33.json"
    # Target number
    TargetFin = 4
    # Gesture continue time
    TimeFin = 5


    # # Hand area in frame
    # hand_width = 200
    # hand_height = 200
    # # default 0, the frame in center, "-" up and left, "+" down and right
    # hand_move_h = 0     # horizontal
    # hand_move_v = 0   # vertical
    # # Object area in frame
    # obj_width = 200
    # obj_height = 200
    # # default 0, the frame in center, "-" up and left, "+" down and right
    # obj_move_h = 0      # horizontal
    # obj_move_v = 0   # vertical
    
    '''
    # Image hand_image.jpg parameter
    # Hand area in frame
    hand_width = 250
    hand_height = 250
    # default 0, the frame in center, "-" up and left, "+" down and right
    hand_move_h = 120     # horizontal
    hand_move_v = 0     # vertical
    # Object area in frame
    obj_width = 250
    obj_height = 250
    # default 0, the frame in center, "-" up and left, "+" down and right
    obj_move_h = -120      # horizontal
    obj_move_v = -70      # vertical
    '''
    '''
    # Video hand_video.mp4 parameter
    # Hand area in frame
    hand_width = 350
    hand_height = 350
    # default 0, the frame in center, "-" up and left, "+" down and right
    hand_move_h = 0         # horizontal
    hand_move_v = 175       # vertical
    # Object area in frame
    obj_width = 400
    obj_height = 300
    # default 0, the frame in center, "-" up and left, "+" down and right
    obj_move_h = 0          # horizontal
    obj_move_v = -150       # vertical
    '''
    '''
    # URL parameter
    # Hand area in frame
    hand_width = 200
    hand_height = 200
    # default 0, the frame in center, "-" up and left, "+" down and right
    hand_move_h = 0         # horizontal
    hand_move_v = 100       # vertical
    # Object area in frame
    obj_width = 250
    obj_height = 150
    # default 0, the frame in center, "-" up and left, "+" down and right
    obj_move_h = 0          # horizontal
    obj_move_v = -50       # vertical
    '''
    # WEBCAM parameter
    # Hand area in frame
    hand_width = 250
    hand_height = 250
    # default 0, the frame in center, "-" up and left, "+" down and right
    hand_move_h = -175         # horizontal
    hand_move_v = 0       # vertical
    # Object area in frame
    obj_width = 250
    obj_height = 250
    # default 0, the frame in center, "-" up and left, "+" down and right
    obj_move_h = 175          # horizontal
    obj_move_v = 0       # vertical

    # control parameter
    detecthand = True      # Detect hand
    detectobject = True   # Detect object
    outputToFile = False    # output the predicted result to image or video file
    displayScreen = True   # Do you want to show the image on LCD?

    
