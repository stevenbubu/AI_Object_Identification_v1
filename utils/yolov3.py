#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import YOLOv3.core.utils as utils
import tensorflow as tf
from PIL import Image
import utils.func_util as func
import datetime

def yolov3(pb_file, image_path, recordpath, option):
    return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
    # pb_file         = "./yolov3_coco.pb"
    # image_path      = "./docs/images/hand_image.jpg"
    num_classes     = 80
    input_size      = 416
    graph           = tf.Graph()
    config          = tf.ConfigProto()
    
    gpu_id = int(option["gpu"])
    gpu_memory_fraction = float(option["mem_percent"])
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        except RuntimeError as e:
            print("GPU setting error. Msg:" + e)


    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_size = original_image.shape[:2]
    image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...]

    return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


    with tf.Session(config=config, graph=graph) as sess:
        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image, label, score = utils.draw_bbox(original_image, bboxes)
    func.save_txt(recordpath, string="\n" + str(datetime.datetime.now()))
    for i in range(len(label)):
        func.save_txt(recordpath, str(label[i])+" "+ str(score[i]), string="\n")
        print(str(label[i])+" "+ str(score[i]))
    func.save_txt(recordpath, string="\n")
    # image = Image.fromarray(image)
    # image.show()




