# AI Object Identification System
## Opencv + Yolo_v3

**在linux建立虛擬環境**

cd ~/Desktop
1. 首先建立python3.6的虛擬環境
  `python3 -m venv work/wins`
2. 啟動虛擬環境
`source work/wins/bin/activate`
3. pip安裝需要的套件
`pip install -r AI_identification_v1/requirement.txt`

**準備訓練Darknet模型**

在AI_identification_v1下建立Image資料夾，在Image下再建立images、labels資料夾。把準備好的圖片放在images資料夾，把voc格式的label放在labels資料夾。

4. 執行1_labels_to_yolo_format.py把label轉換成yolo專用格式。
`python AI_identification_v1/yolov3_608/1_labels_to_yolo_format.py`
5. 把資料分成train、test資料夾
`python AI_identification_v1/yolov3_608/2_split_train_test.py`
6. 建立config file
`python AI_identification_v1/yolov3_608/3_make_cfg_file.py`
7. Training
`python AI_identification_v1/yolov3_608/4_train_yolo.py`
8. 查看訓練情況
`python AI_identification_v1/yolov3_608/visualize_loss_iou.py`
9. gpu-status.sh 可以看gpu使用狀況
`./gpu-status.sh`

**執行判斷手勢及目標物功能**

* 使用在圖片
`python AI_identification_v1/getsignal/getsignal.py -i AI_identification_v1/getsignal/data/4.jpg`
* 使用在影片
`python AI_identification_v1/getsignal/getsignal.py -v AI_identification_v1/getsignal/data/bright_far_coplx.mp4`
* 使用在url link
`python AI_identification_v1/getsignal/getsignal.py -s https://www.youtube.com/watch?v=omhjl-ia42Q&list=PLWA6_OKb3JSBen6SSJ6KZkz4w4gcOy0Bz&index=10`

**系統架構**

![](https://i.imgur.com/puWF3eK.jpg)





