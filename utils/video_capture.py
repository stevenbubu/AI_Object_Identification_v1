import pafy, ffmpeg
import math
import numpy as np
import subprocess
import cv2

import logging.config
import config.log_config as logcfg
logging.config.dictConfig(logcfg.config)
logger = logging.getLogger("StreamLogger")


def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    # 'streams': {'index': 0, 'codec_name': 'vp8', 'codec_long_name': 'On2 VP8', 'profile': '0', 
    # 'codec_type': 'video', 'codec_time_base': '100/2997', 'codec_tag_string': '[0][0][0][0]', 
    # 'codec_tag': '0x0000', 'width': 480, 'height': 360, 'coded_width': 480, 'coded_height': 360, 
    # 'has_b_frames': 0, 'sample_aspect_ratio': '1:1', 'display_aspect_ratio': '4:3', 
    # 'pix_fmt': 'yuv420p', 'level': -99, 'field_order': 'progressive', 'refs': 1, 
    # 'r_frame_rate': '2997/100', 'avg_frame_rate': '2997/100', 'time_base': '1/1000', 
    # 'start_pts': 0, 'start_time': '0.000000', 'disposition': {'default': 1, 'dub': 0, 
    # 'original': 0, 'comment': 0, 'lyrics': 0, 'karaoke': 0, 'forced': 0, 'hearing_impaired': 0, 
    # 'visual_impaired': 0, 'clean_effects': 0, 'attached_pic': 0, 'timed_thumbnails': 0}}
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height


def start_ffmpeg_process1(in_filename):
    logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height):
    logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
        .output(out_filename, pix_fmt='yuv420p')
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)


def read_frame(process1, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame


def write_frame(process2, frame):
    logger.debug('Writing frame')
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )


def VideoCapture(in_filename=None, out_filename=None):
    # url = 'https://youtu.be/W1yKqFZ34y4'
    # url = 'https://www.youtube.com/watch?v=BZP1rYjoBgI'
    # url = 'https://www.youtube.com/watch?v=8tpxwFEIxi0'
    # url = 'https://www.youtube.com/watch?v=DGQm5tbpUyE'
    # vPafy = pafy.new(url)
    # play = vPafy.getbest()
    # in_filename = play.url
    width, height = get_video_size(in_filename)
    
    process1 = start_ffmpeg_process1(in_filename)
    if out_filename is not None:
        process2 = start_ffmpeg_process2(out_filename, width, height)
              
    if process1:
        loop = True
        while loop:              
            in_frame = read_frame(process1, width, height)
            if in_frame is None:
                logger.info('End of input stream')
                break
            
            cv2.imshow("OpenCV",in_frame.astype(np.uint8))  
            if cv2.waitKey(1000) == ord('q'): break
                      
            if out_filename is not None:
                write_frame(process2, in_frame)
        else:
            loop = False
            logger.info('while Done')
         
    logger.info('Waiting for ffmpeg process1')
    process1.wait()

    if out_filename is not None:
        logger.info('Waiting for ffmpeg process2')
        process2.stdin.close()
        process2.wait()

    logger.info('Done')


# VideoCapture(in_filename = '/home/jdwei/Desktop/convolutional-pose-machines/test_imgs/hand3.jpg',
#             out_filename = '/home/jdwei/Desktop/convolutional-pose-machines/test_imgs/output.jpg')