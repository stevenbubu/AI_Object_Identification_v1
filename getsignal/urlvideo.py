import subprocess, os, sys
import argparse
import getsignal_lib as lib
import pafy, vlc
import time
from urllib.parse import urlparse
import cv2

parser = argparse.ArgumentParser(description="Only accept url string.")
parser.add_argument("-s", dest='url', action='store', help='url video stream')

args = parser.parse_args()
scheme, netloc, path, params, query, fragment = urlparse(args.url)
if "http" not in scheme:
    print("Input url ", args.url, " doesn't exist")
    sys.exit(1)

if (args.url):

    video = pafy.new(args.url)
    snapFile = sys.path[0] + '/data/urlvideo/urlvideo_snap.png'
    best = video.getbest()

    Instance = vlc.Instance('--input-repeat=-1', '--fullscreen', '--mouse-hide-timeout=0')
    player = Instance.media_player_new()
    media = Instance.media_new(best.url)
    media.get_mrl()
    player.set_media(media)
    player.play()
    start = time.time()
    playing = set([1,2,3,4])
    
    state = player.get_state()
    
    # if player.is_playing():
    #     lib.get_player_para(player)
    while state in playing: 
        if((time.time() - start) >= video.length):
            player.release()
            sys.exit(1)
        else:
            start1 = time.time()
            player.video_take_snapshot(0, snapFile, 0, 0)
            time.sleep(0.55)  # Make detect time near 1 seconds          
            executeCmd = "python3 " + sys.path[0] + "/urlimage.py" + " -i " +  snapFile
            lib.call_pyfile(executeCmd)
            print("Pass {:.3f} seconds".format(time.time() - start1))

    player.release()  

else:
    print("Not the correct input.")
    sys.exit(1)
