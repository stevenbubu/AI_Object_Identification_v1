import glob, os, sys
import os.path
import time
from shutil import copyfile
import cv2
from xml.dom import minidom
from os.path import basename
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#--------------------------------------------------------------------
folderCharacter = "/"  # \\ is for windows
filedir = os.path.abspath(os.path.join(sys.path[0],os.path.pardir)) # parent dir
xmlFolder = filedir + "/Image/labels"
imgFolder = filedir + "/Image/images"
negFolder = filedir + "/Image/negatives"
saveYoloPath = filedir + "/Image/yolo"
classList = { "aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4, "bus":5, "car":6, \
"cat":7, "chair":8, "cow":9, "diningtable":10, "dog":11, "horse":12, "motorbike":13, \
"person":14, "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19 }

#------------------------------------------------------------ 

if not os.path.exists(saveYoloPath):
    os.makedirs(saveYoloPath)

def transferYolo(xmlFilepath, imgFilepath, labelGrep=""):
    global imgFolder

    img_file, img_file_extension = os.path.splitext(imgFilepath)
    img_filename = basename(img_file)
    print(imgFilepath)

    if(xmlFilepath is not None):
        img = cv2.imread(imgFilepath)
        imgShape = img.shape
        print (img.shape)
        img_h = imgShape[0]
        img_w = imgShape[1]

        labelXML = minidom.parse(xmlFilepath)
        labelName = []
        labelXmin = []
        labelYmin = []
        labelXmax = []
        labelYmax = []
        labelindex = []
        totalW = 0
        totalH = 0
        countLabels = 0
        
        tmpArrays = labelXML.getElementsByTagName("filename")
        for elem in tmpArrays:
            filenameImage = elem.firstChild.data

        tmpArrays = labelXML.getElementsByTagName("name")
        for elem in tmpArrays:
            if str(elem.firstChild.data) in classList.keys():
                labelName.append(str(elem.firstChild.data))
                labelindex.append(int(tmpArrays.index(elem)))

        tmpArrays = labelXML.getElementsByTagName("xmin")
        for idx in labelindex:
            labelXmin.append(int(float(tmpArrays[idx].firstChild.data)))

        tmpArrays = labelXML.getElementsByTagName("ymin")
        for idx in labelindex:
            labelYmin.append(int(float(tmpArrays[idx].firstChild.data)))

        tmpArrays = labelXML.getElementsByTagName("xmax")
        for idx in labelindex:
            labelXmax.append(int(float(tmpArrays[idx].firstChild.data)))

        tmpArrays = labelXML.getElementsByTagName("ymax")
        for idx in labelindex:
            labelYmax.append(int(float(tmpArrays[idx].firstChild.data)))
            
        yoloFilename = saveYoloPath + folderCharacter + img_filename + ".txt"
        print("writeing to {}".format(yoloFilename))

        with open(yoloFilename, 'a') as the_file:
            i = 0
            for className in labelName:
                if(className==labelGrep or labelGrep==""):
                    classID = classList[className]
                    x = (labelXmin[i] + (labelXmax[i]-labelXmin[i])/2) * 1.0 / img_w 
                    y = (labelYmin[i] + (labelYmax[i]-labelYmin[i])/2) * 1.0 / img_h
                    w = (labelXmax[i]-labelXmin[i]) * 1.0 / img_w
                    h = (labelYmax[i]-labelYmin[i]) * 1.0 / img_h

                    the_file.write(str(classID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
                    i += 1

    else:
        yoloFilename = saveYoloPath + folderCharacter + img_filename + ".txt"
        print("writeing negative file to {}".format(yoloFilename))

        with open(yoloFilename, 'a') as the_file:
            the_file.write('')

    the_file.close()

#---------------------------------------------------------------
fileCount = 0

for file in os.listdir(imgFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
        imgfile = imgFolder + folderCharacter + file
        xmlfile = xmlFolder + folderCharacter + filename + ".xml"

        if(os.path.isfile(xmlfile)):
            print("id:{}".format(fileCount))
            print("processing {}".format(imgfile))
            print("processing {}".format(xmlfile))
            fileCount += 1
            transferYolo( xmlfile, imgfile, "")
            copyfile(imgfile, saveYoloPath + folderCharacter + file)

# for file in os.listdir(negFolder):
#     filename, file_extension = os.path.splitext(file)
#     file_extension = file_extension.lower()
#     imgfile = negFolder + folderCharacter + file

#     if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
#         transferYolo( None, imgfile, "")
#         copyfile(imgfile, saveYoloPath + folderCharacter + file)
