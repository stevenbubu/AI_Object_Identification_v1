# AI Object Identification System v1.1
## Opencv + Google Vision API

**在linux建立虛擬環境**

cd ~/Desktop
1. 首先建立python3.6的虛擬環境
  `python3 -m venv work/wins`
2. 啟動虛擬環境
`source work/wins/bin/activate`
3. pip安裝需要的套件
`pip install -r AI_gesture_obj/requirement.txt`

**Google Vision API Setup**

1. Create a project and enable vision api

    https://console.cloud.google.com/home...

    Get API key and service account key

2. Install following:

    `sudo apt-get update && sudo apt-get install google-cloud-sdk`

3. Log into your google account: 

    `gcloud auth login`
    
Reference： [Google Vision API Setup Python Tutorial - Updated for 2019](https://www.youtube.com/watch?v=bHkQb3gnSRA)

**執行判斷手勢及目標物功能**

* 設定檔在gesture_config.py設定，Input 可為 Image、Video、URL、Webcam

**系統架構**

![](https://i.imgur.com/ypUUVZ1.png)






