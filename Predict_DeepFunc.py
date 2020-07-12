#-*- encoding:utf8 -*-

import fire
import os
import time
import math

import pickle
import pandas as pd
import numpy as np
import torch
# from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torch.nn.init import xavier_normal,xavier_normal_
from torch import nn

from utils.config import DefaultConfig
from models.DeepFunc_model import DeepFunc
from data import data_generator
from utils.evaluation import  compute_mcc, compute_roc, compute_performance



configs = DefaultConfig()
thresholds = configs.thresholds

path_dir = "./saved_model/DeepFunc"

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        xavier_normal_(m.weight.data)
    elif isinstance(m,nn.Linear):
        xavier_normal_(m.weight.data)
    
def predict_func(gpu_tag, model, loader,class_tag,saved_path):

    # Model on eval mode
    model.eval()
    length = len(loader)
    all_label_file = './data/base_data/{0}.csv'.format(class_tag)
    all_labels = np.array(pd.read_csv(all_label_file)['functions'])
    predict_dict = {}
    
    result = []
    for batch_idx, (input,embed_input,interpro_info,entrys) in enumerate(loader):
        with torch.no_grad():
            if gpu_tag and torch.cuda.is_available():
                input_var = torch.autograd.Variable(input.cuda())
                embed_input_var = torch.autograd.Variable(embed_input.cuda())
                interpro_info_var = torch.autograd.Variable(interpro_info.cuda())
                
            else:
                input_var = torch.autograd.Variable(input)
                embed_input_var = torch.autograd.Variable(embed_input)
                interpro_info_var = torch.autograd.Variable(interpro_info)
                
        # compute output
        output = model(input_var, embed_input_var, interpro_info_var)
        predict_bin = output.data.cpu().numpy()

        for ii,entry in enumerate(entrys):

            preds = np.where(predict_bin[ii]>=thresholds[class_tag],1,0)
            predict_dict[entry] = [all_labels[index] for index,k in enumerate(preds) if k == 1]

    save_file = saved_path+'/result_{0}.txt'.format(class_tag)
    with open(save_file,'w') as fw:
        for entry,functions in predict_dict.items():
            fw.write(entry+"  predicted functions:"+'\n')
            if len(functions) == 0:
                fw.write('\n')
            for fun in functions:
                fw.write(entry+"\t"+fun+'\t'+class_tag+'\n')
            fw.write('\n')
            
def predict(predicted_file,saved_path):

    
    embeddings_file = 'utils/embeddings.npy'
    # parameters
    batch_size = configs.batch_size

    predict_dataSet = data_generator.deepFuncDataSet(predicted_file,embeddings_file)

    predict_loader = torch.utils.data.DataLoader(predict_dataSet, batch_size=3, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=1,drop_last=False)

    # Models
    interPro_size = configs.interPro_size
    gpu_tag = 1 
    for class_tag in ['mf','bp','cc']:
        model_file = "{1}/DeepFunc_{0}_model.dat".format(class_tag,path_dir)
        print(model_file)

        class_nums = configs.class_nums[class_tag]
        model = DeepFunc(class_nums)
        model.load_state_dict(torch.load(model_file))
        if gpu_tag and torch.cuda.is_available():
            model = model.cuda()
        print("predict_{0}".format(class_tag))
        predict_func(gpu_tag, model, predict_loader, class_tag,saved_path)

    print('Done!')

if __name__ == '__main__':

    

    predicted_file = './test.txt'
    saved_path = './output_path'
    
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    
    predict(predicted_file,saved_path)

