# -*- coding: utf-8 -*-
# @Time    : 2018/12/30 16:26
# @Author  : lazerliu
# @File    : vis_yolov3_log.py
# @Func    :yolov3 訓練日誌可視化，把該腳本和日誌文件放在同一目錄下運行。

import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

# ==================可能需要修改的地方=====================================#
g_log_path = "train_yolov3.log"  # 此處修改爲你的訓練日誌文件名
# ==========================================================================#

def extract_log(log_file, new_log_file, key_word):
    '''
    :param log_file:日誌文件
    :param new_log_file:挑選出可用信息的extract_log(log_file, new_log_file, key_word):日誌文件
    :param key_word:根據關鍵詞提取日誌信息
    :return:
    '''
    with open(log_file, "r") as f:
        with open(new_log_file, "w") as train_log:
            for line in f:
                # 去除多gpu的同步log
                if "Syncing" in line:
                    continue
                # 去除nan log
                if "nan" in line:
                    continue
                if key_word in line:
                    train_log.write(line)
    f.close()
    train_log.close()


def drawAvgLoss(loss_log_path):
    '''
    :param loss_log_path: 提取到的loss日誌信息文件
    :return: 畫loss曲線圖
    '''
    line_cnt = 0
    for count, line in enumerate(open(loss_log_path, "rU")):
        line_cnt += 1
    result = pd.read_csv(loss_log_path, skiprows=[iter_num for iter_num in range(line_cnt) if ((iter_num < 500))],
                         error_bad_lines=False,
                         names=["loss", "avg", "rate", "seconds", "images"])
    result["avg"] = result["avg"].str.split(" ").str.get(1)
    result["avg"] = pd.to_numeric(result["avg"])

    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(result["avg"].values, label="Avg Loss", color="#ff7043")
    ax.legend(loc="best")
    ax.set_title("Avg Loss Curve")
    ax.set_xlabel("Batches")
    ax.set_ylabel("Avg Loss")
    plt.savefig('Avg_Loss.png')


def drawIOU(iou_log_path):
    '''
    :param iou_log_path: 提取到的iou日誌信息文件
    :return: 畫iou曲線圖
    '''
    line_cnt = 0
    for count, line in enumerate(open(iou_log_path, "rU")):
        line_cnt += 1
    result = pd.read_csv(iou_log_path, skiprows=[x for x in range(line_cnt) if (x % 39 != 0 | (x < 5000))],
                         error_bad_lines=False,
                         names=["Region Avg IOU", "Class", "Obj", "No Obj", "Avg Recall", "count"])
    result["Region Avg IOU"] = result["Region Avg IOU"].str.split(": ").str.get(1)

    result["Region Avg IOU"] = pd.to_numeric(result["Region Avg IOU"])

    result_iou = result["Region Avg IOU"].values
    # 平滑iou曲線
    for i in range(len(result_iou) - 1):
        iou = result_iou[i]
        iou_next = result_iou[i + 1]
        if abs(iou - iou_next) > 0.2:
            result_iou[i] = (iou + iou_next) / 2

    fig = plt.figure(2, figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(result_iou, label="Region Avg IOU", color="#ff7043")
    ax.legend(loc="best")
    ax.set_title("Avg IOU Curve")
    ax.set_xlabel("Batches")
    ax.set_ylabel("Avg IOU")
    plt.savefig('Avg_IOU.png')


if __name__ == "__main__":
    loss_log_path = "train_log_loss.txt"
    iou_log_path = "train_log_iou.txt"

    while True:
        if os.path.exists(g_log_path) is False:
            exit(-1)
        # if os.path.exists(loss_log_path) is False:
        extract_log(g_log_path, loss_log_path, "images")
        # if os.path.exists(iou_log_path) is False:
        extract_log(g_log_path, iou_log_path, "IOU")
        drawAvgLoss(loss_log_path)
        drawIOU(iou_log_path)
        plt.close('all')
        # plt.show()

        # Image display
        img = cv2.imread('Avg_Loss.png')
        cv2.namedWindow('Avg Loss Curve', cv2.WINDOW_NORMAL)
        cv2.imshow('Avg Loss Curve', img.copy())
        
        img2 = cv2.imread('Avg_IOU.png')
        cv2.namedWindow('Region Avg IOU', cv2.WINDOW_NORMAL)
        cv2.imshow('Region Avg IOU', img2.copy())
        
        cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        
    cv2.destroyAllWindows()