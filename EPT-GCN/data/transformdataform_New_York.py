# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:44:04 2021

@author: mofan
"""

import argparse
import os
import numpy as np
import math
import sys

USERNUMBER=3286
data_path='./new-york'


def trans_file(path, trans_path):
    f=open(path)
    f_trans=open(trans_path,'w')
    user_list=[[] for i in range(USERNUMBER)]
    for line in f.readlines():
        u_id, p_id,_=line.split()
        user_list[int(u_id)].append(p_id)
    for i in range(USERNUMBER):
        print(str(i),' '.join(str(user_list[i][j]) for j in range(len(user_list[i]))), file=f_trans)
        f_trans.flush()
    f_trans.close()
    return user_list


if __name__ == '__main__':
    train_path=data_path+'/New_York_train.txt'
    tune_path=data_path+'/New_York_tune.txt'
    test_path=data_path+'/New_York_test.txt'
    trans_train_path=data_path+'/New_York_train_trans.txt'
    trans_tune_path=data_path+'/New_York_tune_trans.txt'
    trans_test_path=data_path+'/New_York_test_trans.txt'
    
    
    user_list_train=trans_file(train_path, trans_train_path)
    user_list_tune=trans_file(tune_path, trans_tune_path)
    user_list_test=trans_file(test_path, trans_test_path)
    