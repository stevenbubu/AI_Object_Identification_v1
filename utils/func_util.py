import os
import logging as log


def timing(sec):
    day = int(sec//(60*60*24));  day_rm = sec%(60*60*24)
    hour = int(day_rm//(60*60)); hour_rm = day_rm%(60*60)
    mins = int(hour_rm//60);     sec = hour_rm%60

    day = str("0"+str(day)) if day<10 else str(day)
    hour = str("0"+str(hour)) if hour<10 else str(hour)
    mins = str("0"+str(mins)) if mins<10 else str(mins)
    sec = str("0"+str(sec)) if sec<10 else str("%.2f"%sec)
    string = str(day) + ":" + str(hour) + ":" + str(mins) + ":" + str(sec)
    log.info("Time: " + string)
    return string


def createFolder(folders, path=None):
    for folder in folders:
        if path is not None:
            fdpath = os.path.join(path, folder)    
        else:
            fdpath = folder
        try:
            if not os.path.exists(fdpath):
                os.makedirs(fdpath)
            else:
                log.info('Folder already exists: ' +  folder)
        except OSError:
             log.info('Error: Creating directory. ' + fdpath)


def save_txt(txt_path, txt_content, string=""):
    with open(txt_path, 'a') as txtfile:
        txtfile.write(string + txt_content)   

