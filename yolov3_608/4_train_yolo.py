import os, time, sys
import subprocess
from shutil import copyfile

filedir = os.path.abspath(os.path.join(sys.path[0],os.path.pardir)) # parent dir
cfgFolder = filedir + "/Image/cfg.voc608"

modelYOLO = "yolov3"  #yolov3 or yolov3-tiny
classList = { "aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4, "bus":5, "car":6, \
"cat":7, "chair":8, "cow":9, "diningtable":10, "dog":11, "horse":12, "motorbike":13, \
"person":14, "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19 }
folderCharacter = "/"  # \\ is for windows

# copy empty cfg files to dest
defcfgdir = sys.path[0] + "/cfg/" # empty cfg file
# r=root, d=directories, f = files
for r, d, f in os.walk(defcfgdir):
    for file in f:
        copyfile(os.path.join(r, file), os.path.join(cfgFolder + folderCharacter, file))

numBatch = 64
numSubdivision = 16
darknetEcec = filedir + "/darknet/darknet"

#------------------------------------------------------

def downloadPretrained(url):
    import wget
    print("Downloading the pretrained model darknet53.conv.74, please wait.")
    wget.download(url, cfgFolder + folderCharacter)

if not os.path.exists(cfgFolder + folderCharacter + "darknet53.conv.74"):
    downloadPretrained("https://pjreddie.com/media/files/darknet53.conv.74")

classNum = len(classList)
filterNum = (classNum + 5) * 3

if(modelYOLO == "yolov3"):
    fileCFG = "yolov3.cfg"

else:
    fileCFG = "yolov3-tiny.cfg"

with open(cfgFolder + folderCharacter + fileCFG) as file:
    file_content = file.read()

file.close

file_updated = file_content.replace("{BATCH}", str(numBatch))
file_updated = file_updated.replace("{SUBDIVISIONS}", str(numSubdivision))
file_updated = file_updated.replace("{FILTERS}", str(filterNum))
file_updated = file_updated.replace("{CLASSES}", str(classNum))

file = open(cfgFolder + folderCharacter + fileCFG, "w")
file.write(file_updated)
file.close

if not os.path.exists(cfgFolder + folderCharacter + "weights" + folderCharacter + "yolov3.backup"):
    executeCmd = darknetEcec + " detector train " + cfgFolder + folderCharacter + \
        "obj.data " + cfgFolder + folderCharacter + fileCFG + " " + cfgFolder + \
            folderCharacter + "darknet53.conv.74" + " -gpus 1 | tee train_yolov3.log"

else:
    executeCmd = darknetEcec + " detector train " + cfgFolder + folderCharacter + \
            "obj.data " + cfgFolder + folderCharacter + fileCFG + " " + cfgFolder + \
                folderCharacter + "weights" + folderCharacter + "yolov3.backup" + " -gpus 1 | tee train_yolov3.log"        

print("Execute darknet training command:")
print("    " + executeCmd)

subprocess.Popen(executeCmd, shell=True)
