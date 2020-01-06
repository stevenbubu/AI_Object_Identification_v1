# AI Object Identification System v1.1
## OpenCV + Google Vision API

**I、在linux建立虛擬環境**
1. 首先建立python3.6的虛擬環境
    `cd ~/Desktop/work`
    `python3 -m venv Detect`
3. 啟動虛擬環境
    `source Detect/bin/activate`
    `pip install –upgrade pip`
3. pip安裝需要的套件
    `cd ~/Desktop/AI_identification_v1.1`
    `pip install -r requirement.txt`

**II、Google Vision API Setup**

1. Create a project and enable vision api

    https://console.cloud.google.com/home...

    Get API key and service account key

2. Install following:

    `sudo apt-get update && sudo apt-get install google-cloud-sdk`

3. Log into your google account: 

    `gcloud auth login`
    
Reference： [Google Vision API Setup Python Tutorial - Updated for 2019](https://www.youtube.com/watch?v=bHkQb3gnSRA)

**III、判斷手勢及目標物**

1. 在 config/gesture.config 設定參數，MediaType可以為IMAGE、VIDEO、URL、WEBCAM

**IV、系統架構**

![](https://i.imgur.com/ypUUVZ1.png)






