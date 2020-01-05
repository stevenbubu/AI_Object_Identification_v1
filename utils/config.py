import tensorflow as tf
import string
import logging as log
import sys
import pynvml
pynvml.nvmlInit()

'''
For config file filter
'''
def _get_config_from_pipeline_file(pipeline_config_path):
    config = {}
    key_list = []
    level = -1
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        proto_str = proto_str.replace(" ","").replace(",","").split("\n")
        for item in proto_str:
            if (item.find('#') == -1) and (item.find('\'\'\'') == -1):
                tmp_dict = {}
                tmp_dict = config
                if item.find('{') != -1:
                    level += 1
                    tmp_item = item.replace(":","").replace("{","")
                    for _ in range(level):
                        for k, v in tmp_dict.items():
                            tmp_dict = v
                    tmp_dict.update({tmp_item:dict()})
                    key_list.append(tmp_item)        
                elif item.find('}') != -1:
                    level -= 1
                    key_list.pop()
                else:
                    if len(item.split(":")) == 2:
                        tmp_dict = config
                        for _ in range(level):
                            for k, v in tmp_dict.items():
                                tmp_dict = v
                        tmp_dict = tmp_dict.get(key_list[level])
                        try:
                            if item.split(":")[1] in ["true", "True", "TRUE"]:
                                tmp_dict.update({item.split(":")[0]:bool(True)})
                            elif item.split(":")[1] in ["false", "False", "FALSE"]:
                                tmp_dict.update({item.split(":")[0]:bool(False)})
                            else:
                                tmp_dict.update({item.split(":")[0]:float(item.split(":")[1])})
                        except:
                            tmp_dict.update({item.split(":")[0]:str(item.split(":")[1])})
                    elif len(item.split(":")) > 2 :
                        tmp_dict = config
                        for _ in range(level):
                            for k, v in tmp_dict.items():
                                tmp_dict = v
                        tmp_dict = tmp_dict.get(key_list[level])
                        if item.split(":")[1] in ['http','https']:
                            tmp_dict.update({item.split(":")[0]:":".join(item.split(":")[1:])})
    return config


def get_config_from_init_config(init_configs_path, pipeline_config_path="config/pipeline.config"):
    tf.io.gfile.copy(init_configs_path, pipeline_config_path, overwrite=True)
    return _get_config_from_pipeline_file(pipeline_config_path)

