# AI Object Identification System v1.0
## OpenCV + YOLOv3

**I、在linux建立虛擬環境**
1. 首先建立python3.6的虛擬環境
    `cd ~/Desktop/work`
    `python3 -m venv Detect`
3. 啟動虛擬環境
    `source Detect/bin/activate`
    `pip install –upgrade pip`
3. pip安裝需要的套件
    `cd ~/Desktop/AI_identification_v1`
    `pip install -r requirement.txt`

**II、判斷手勢及目標物**
1. 在 config/gesture.config 設定參數，MediaType可以為IMAGE、VIDEO、URL、WEBCAM

**III、Reference**
1. [pure tensorflow Implement of YOLOv3 with support to train your own dataset](https://github.com/YunYang1994/tensorflow-yolov3)

**IV、系統架構**

![](https://i.imgur.com/puWF3eK.jpg)





