import os, sys

#-------------------------------------------------------------
classes = 20

#Same with you defined in 1_labels_to_yolo_format.py
classList = { "aeroplane":0, "bicycle":1, "bird":2, "boat":3, "bottle":4, "bus":5, "car":6, \
"cat":7, "chair":8, "cow":9, "diningtable":10, "dog":11, "horse":12, "motorbike":13, \
"person":14, "pottedplant":15, "sheep":16, "sofa":17, "train":18, "tvmonitor":19 }

folderCharacter = "/"  # \\ is for windows
filedir = os.path.abspath(os.path.join(sys.path[0],os.path.pardir)) # parent dir
cfgFolder = filedir + "/Image/cfg.voc608"

#-------------------------------------------------------------

cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"

if not os.path.exists(cfgFolder + folderCharacter + "weights"):
    os.makedirs(cfgFolder + folderCharacter + "weights")
    print("all weights will generated in here: " + cfgFolder + folderCharacter + "weights" + folderCharacter)


with open(cfgFolder + folderCharacter + cfg_obj_data, 'w') as the_file:
    the_file.write("classes= " + str(classes) + "\n")
    the_file.write("train  = " + cfgFolder + folderCharacter + "train.txt\n")
    the_file.write("valid  = " + cfgFolder + folderCharacter + "test.txt\n")
    the_file.write("names = " + cfgFolder + folderCharacter + "obj.names\n")
    the_file.write("backup = " + cfgFolder + folderCharacter + "weights/")

the_file.close()

print("and cfg folder: " + cfgFolder + " ,is ready for training.")

with open(cfgFolder + folderCharacter + cfg_obj_names, 'w') as the_file:
    for className in classList:
        the_file.write(className + "\n")

the_file.close()

