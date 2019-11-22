import subprocess, sys, os
import math
import numpy as np
import cv2
import pyautogui, time
import getsignal_lib as lib


_COLOR_RED   = (0, 0, 255)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE  = (255, 0, 0)

def _round_int(value):
    result = int(np.rint(value))
    return result

# 移除视频数据的背景噪声
def _remove_background(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2() # 利用BackgroundSubtractorMOG2算法消除背景
    fgmask = fgbg.apply(frame)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# 视频数据的人体皮肤检测
def _bodyskin_detetc(frame):
    # 肤色检测: YCrCb之Cr分量 + OTSU二值化
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb) # 分解为YUV图像,得到CR分量
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0) # 高斯滤波
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # OTSU图像二值化
    return skin

# 检测图像中的凸点(手指)个数
def _get_contours(array):
    # 利用findContours检测图像中的轮廓, 其中返回值contours包含了图像中所有轮廓的坐标点
    _, contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _get_eucledian_distance(a, b):
    distance = math.sqrt( (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return distance

# 根据图像中凹凸点中的 (开始点, 结束点, 远点)的坐标, 利用余弦定理计算两根手指之间的夹角, 其必为锐角, 根据锐角的个数判别手势.
def _get_defects_count(array, contour, defects, verbose = False):
    ndefects = 0
    for i in range(defects.shape[0]):
        s,e,f,_ = defects[i,0]
        beg     = tuple(contour[s][0])
        end     = tuple(contour[e][0])
        far     = tuple(contour[f][0])
        a       = _get_eucledian_distance(beg, end)
        b       = _get_eucledian_distance(beg, far)
        c       = _get_eucledian_distance(end, far)
        angle   = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) #* 57
        if angle <= math.pi/2 : #90:
            ndefects = ndefects + 1
            if verbose:
                cv2.circle(array, far, 6, _COLOR_GREEN, 2)
        if verbose:
            cv2.line(array, beg, end, _COLOR_RED, 2)

    return array, ndefects

def _get_tip_position(array, contour, verbose = False):
    approx_contour = cv2.approxPolyDP(contour, 0.08 * cv2.arcLength(contour, True), True)
    convex_points  = cv2.convexHull(approx_contour, returnPoints = True)
    cx, cy     = 999, 999

    oneflag = 0 # for single finger
    if len(convex_points) == 3:
        oneflag = 1

    for point in convex_points:
        cur_cx, cur_cy = point[0][0], point[0][1]

        if verbose:
            cv2.circle(array, (cur_cx, cur_cy), 8, _COLOR_BLUE, 2)
            
        if (cur_cy < cy):
            cx, cy = cur_cx, cur_cy
    
    (screen_x, screen_y) = pyautogui.size()

    height, width = array.shape[0], array.shape[1]
    x = _round_int((float(cx))/(width-0)*(screen_x+1))
    y = _round_int((float(cy))/(height-0)*(screen_y+1))
    return (array, (x, y), oneflag)

def grdetect(array, verbose = False, show = False, hand = False, cnt = None):
    copy        = array.copy()
    array       = _remove_background(array) # 移除背景
    thresh      = _bodyskin_detetc(array)
    contours    = _get_contours(thresh.copy()) # 计算图像的轮廓
    largecont   = max(contours, key = lambda contour: cv2.contourArea(contour))
    hull        = cv2.convexHull(largecont, returnPoints = False) # 计算轮廓的凸点
    defects     = cv2.convexityDefects(largecont, hull) # 计算轮廓的凹点

    if defects is not None:
        # 利用凹陷点坐标, 根据余弦定理计算图像中锐角个数
        copy, ndefects = _get_defects_count(copy, largecont, defects, verbose = verbose)

        if show is True:
            if ndefects == 0:
                copy, tip, onefin  = _get_tip_position(copy, largecont, verbose = verbose)
                if onefin:
                    print(onefin)
                else:
                    print(ndefects)

            elif ndefects == 1:
                print("2")

            elif ndefects == 2:
                print("3")

            elif ndefects == 3:
                print("4")

            elif ndefects == 4:
                print("5")

        elif hand is True and cnt is not None:
            if ndefects == 0:
                copy, tip, onefin  = _get_tip_position(copy, largecont, verbose = verbose)
                if onefin:
                    if cnt[1] == 0:
                        for p in range(len(cnt)):
                            cnt[p] = 0
                    cnt[1] = cnt[1] + 1
                    print(onefin)
                else:
                    if cnt[0] == 0:
                        for p in range(len(cnt)):
                            cnt[p] = 0
                    cnt[0] = cnt[0] + 1
                    print(ndefects)

            elif ndefects == 1:
                if cnt[2] == 0:
                    for p in range(len(cnt)):
                        cnt[p] = 0
                cnt[2] = cnt[2] + 1
                print("2")

            elif ndefects == 2:
                if cnt[3] == 0:
                    for p in range(len(cnt)):
                        cnt[p] = 0
                cnt[3] = cnt[3] + 1
                print("3")

            elif ndefects == 3:
                if cnt[4] == 0:
                    for p in range(len(cnt)):
                        cnt[p] = 0
                cnt[4] = cnt[4] + 1
                print("4")

            elif ndefects == 4:
                if cnt[5] == 0:
                    for p in range(len(cnt)):
                        cnt[p] = 0
                cnt[5] = cnt[5] + 1
                print("5")

    return copy

def call_pyfile(command):

    # python3 /home/jdwei/Desktop/AI_idfy/yolov3_608/playYOLO.py -i /home/jdwei/Desktop/AI_idfy/yolov3_608/2010_004995.jpg
    # executeCmd = "python3 " + parentdir + "/yolov3_608/playYOLO.py" + " -i " +  filedir
    
    p = subprocess.Popen(command, shell=True)

    while p.poll() is None:
        # Process hasn't exited yet, let's wait.
        time.sleep(0.2)
        p.kill()

def frame_border(array, frame_width, frame_height, frame_move_h, frame_move_v):

    width = array.get(cv2.CAP_PROP_FRAME_WIDTH); height = array.get(cv2.CAP_PROP_FRAME_HEIGHT)
    center_x = int(width/2 + frame_move_h); center_y = int(height/2 + frame_move_v)
    up_left_x = int(center_x - frame_width/2)
    up_left_y = int(center_y - frame_height/2)
    btm_right_x = int(up_left_x + frame_width)
    btm_right_y = int(up_left_y + frame_height)

    if up_left_x < 0:
        up_left_x = 0; btm_right_x = up_left_x + frame_width

        if btm_right_x > width:
            btm_right_x = width
        #     print('Frame width = {:d}, smaller than default width: {:d}'.format(int(btm_right_x - up_left_x), int(frame_width)))
        # else:
        #     print('Frame width = {:d}, up_left_x: {:d}, btm_right_x: {:d}'.format(int(btm_right_x - up_left_x), int(up_left_x), int(btm_right_x)))

    elif btm_right_x > width:
        up_left_x = width - frame_width; btm_right_x = width; 

        if up_left_x < 0:
            up_left_x = 0
        #     print('Frame width = {:d}, smaller than default width: {:d}'.format(int(btm_right_x - up_left_x), int(frame_width)))
        # else:
        #     print('Frame width = {:d}, up_left_x: {:d}, btm_right_x: {:d}'.format(int(btm_right_x - up_left_x), int(up_left_x), int(btm_right_x)))
    
    if up_left_y < 0:
        up_left_y = 0; btm_right_y = up_left_y + frame_height

        if btm_right_y > height:
            btm_right_y = height
        #     print('Frame height = {:d}, smaller than default height: {:d}'.format(int(btm_right_y - up_left_y), int(frame_height)))
        # else:
        #     print('Frame height = {:d}, up_left_y: {:d}, btm_right_y: {:d}'.format(int(btm_right_y - up_left_y), int(up_left_y), int(btm_right_y)))

    elif btm_right_y > height:
        up_left_y = height - frame_height; btm_right_y = height;

        if up_left_y < 0:
            up_left_y = 0
        #     print('Frame height = {:d}, smaller than default height: {:d}'.format(int(btm_right_y - up_left_y), int(frame_height)))
        # else:
        #     print('Frame height = {:d}, up_left_y: {:d}, btm_right_y: {:d}'.format(int(btm_right_y - up_left_y), int(up_left_y), int(btm_right_y)))

    # print(int(up_left_x), int(up_left_y), int(btm_right_x), int(btm_right_y))

    return int(up_left_x), int(up_left_y), int(btm_right_x), int(btm_right_y)

def get_player_para(player):
    length = player.get_length()
    width = player.video_get_width()
    size = player.video_get_size()
    title = player.video_get_title_description()
    video_track = player.video_get_track_description()
    audio_track = player.audio_get_track_description()
    fps = player.get_fps()
    rate = player.get_rate()
    track_count = player.video_get_track_count()
    track = player.video_get_track()
    print ('length: ',length)
    print ('width: ',width)
    print ('size: ', size)
    print ('title: ',title)
    print ('video_track: ',video_track)
    print ('audio_track: ',audio_track)
    print ('fps: ',fps)
    print ('rate: ', rate)
    print ('track_count: ',track_count)
    print ('track: ',track)

def DetectObject(txtpath, targetpath, frame, cnt, fingers = 3, times = 5):

    predir = os.path.abspath(os.path.join(sys.path[0],os.path.pardir)) # parent dir
    Targetimgdir = os.path.abspath(targetpath)

    numtmp = [0, 0, 0, 0, 0, 0]
    tmp = 0
    for idx in range(len(cnt)):
        if(cnt[idx] == 1):
            if os.path.isfile(txtpath):
                txtfile = open(txtpath,'r')
                strnum = txtfile.read()
                txtfile.close()
                for i in strnum:
                    if i in [" ", "[", "]", "," ]:
                        strnum = strnum.replace(i, "")
                for i in range(len(strnum)):
                    numtmp[i] = int(strnum[i])
                if numtmp[idx] == 0:
                    numtmp = [0, 0, 0, 0, 0, 0]
                numtmp[idx] += 1

                if (numtmp[fingers] == times):
                    numtmp = [0, 0, 0, 0, 0, 0]
                    cv2.imwrite(targetpath, frame.astype(np.uint8))
                    # Run YOLOv3 detect object
                    executeCmd = "python3 " + predir + "/yolov3_608/playYOLO.py" + " -i " +  Targetimgdir
                    call_pyfile(executeCmd)

                txtfile = open(txtpath,'w+')
                txtfile.write(str(numtmp))
            else:
                txtfile = open(txtpath,'w+')
                txtfile.write(str(cnt))
            txtfile.close()

        tmp += cnt[idx]
        if (idx == (len(cnt)-1)) and (tmp == 0):
            txtfile = open(txtpath,'w+')
            txtfile.write(str(cnt))
            txtfile.close()