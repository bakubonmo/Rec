
# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [2]))

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
 
import torch.nn.functional as F
import torch.autograd as autograd 

import pdb
from collections import defaultdict
import time
import data_utils 
import evaluate
from shutil import copyfile


dataset_base_path='../data/gowalla-geo'  
 
##gowalla
user_num=18737
item_num=32510 
factor_num=64
batch_size=2048*512
top_k=20 
num_negative_test_val=-1##all 
  

run_id="s0"
print(run_id)
dataset='gowalla-geo'
path_save_base='./log/'+dataset+'/newloss'+run_id
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    os.makedirs(path_save_base)   
result_file=open(path_save_base+'/results.txt','w+')#('./log/results_gcmc.txt','w+')
copyfile('./train_gowalla_geo.py', path_save_base+'/train_gowalla_geo'+run_id+'.py')

path_save_model_base='../newlossModel/'+dataset+'/s'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)  

   
training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/datanpy/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/datanpy/testing_set.npy',allow_pickle=True)  
val_user_set,val_item_set,val_set_count = np.load(dataset_base_path+'/datanpy/val_set.npy',allow_pickle=True)    
user_rating_set_all = np.load(dataset_base_path+'/datanpy/user_rating_set_all.npy',allow_pickle=True).item()
[table_geo_u2u] = np.load(dataset_base_path+'/datanpy/training_geo_neighbor_u2u.npy',allow_pickle=True)
[table_geo_i2i] = np.load(dataset_base_path+'/datanpy/training_geo_neighbor_i2i.npy',allow_pickle=True)
[table_geo_distance_user] = np.load(dataset_base_path+'/datanpy/training_geo_user_set.npy',allow_pickle=True)
[table_geo_distance_item] = np.load(dataset_base_path+'/datanpy/training_geo_set.npy',allow_pickle=True)

def KNN(table_geo_distance,topk):
    for i in range(len(table_geo_distance)):
        nonzero_num=np.count_nonzero(table_geo_distance[i])
        if(nonzero_num>topk):
            set_zero_idx=(-table_geo_distance[i]).argsort()[:nonzero_num-topk]
            table_geo_distance[i][set_zero_idx]=0
    return table_geo_distance
table_geo_distance_user=KNN(table_geo_distance_user,20)
table_geo_distance_item=KNN(table_geo_distance_item,20)

def set_reciprocal(table_geo_distance):
    idx_nonzeros=np.where(table_geo_distance!=0)
    table_geo_distance[idx_nonzeros]=1/table_geo_distance[idx_nonzeros]
    maxi=table_geo_distance.max(axis=0)
    idx_nonzeros=np.where(maxi!=0)
    for i in idx_nonzeros[0]:
        table_geo_distance[i]=table_geo_distance[i]/maxi[i]
    return table_geo_distance

table_geo_distance_user=set_reciprocal(table_geo_distance_user)
table_geo_distance_item=set_reciprocal(table_geo_distance_item)
   
def readD(set_matrix,num_):
    user_d=[] 
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d
u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)
#1/(d_i+1)
d_i_train=u_d
d_j_train=i_d
#1/sqrt((d_i+1)(d_j+1)) 
# d_i_j_train=np.sqrt(u_d*i_d) 

#user-item  to user-item matrix and item-user matrix
def readTrainSparseMatrix(set_matrix,is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[] 
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        len_set=len(set_matrix[i])  
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            #1/sqrt((d_i+1)(d_j+1)) 
            user_items_matrix_v.append(d_i_j)#(1./len_set) 
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

sparse_u_i=readTrainSparseMatrix(training_user_set,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,False)

#user-item  to user-item matrix and item-user matrix

def readD_geo(set_matrix,num_):
    user_d=[] 
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i]))  
        user_d.append(len_set)
    return user_d
u_d_geo=readD_geo(table_geo_u2u,user_num)
i_d_geo=readD_geo(table_geo_i2i,item_num)
#1/(d_i+1)
d_i_train_geo=u_d_geo
d_j_train_geo=i_d_geo

def readTrainSparseMatrix_geo(set_matrix,geo_distance,is_user):
    geo_matrix_i=[]
    geo_matrix_v=[] 
    if is_user:
        d_i=u_d_geo
        d_j=u_d_geo
    else:
        d_i=i_d_geo
        d_j=i_d_geo
    for i in set_matrix:
        len_set=len(set_matrix[i])  
        for j in set_matrix[i]:
            i=int(i)
            j=int(j)
            if(i!=j):
                #d_i_j=np.sqrt(d_i[i]*d_j[j])
                d_i_j=np.sqrt(d_i[i])*(geo_distance[i][j])
                #1/sqrt((d_i+1)(d_j+1))               
            else:
                d_i_j=0
            geo_matrix_i.append([i,j])
            geo_matrix_v.append(d_i_j)#(1./len_set) 
    geo_matrix_i=torch.cuda.LongTensor(geo_matrix_i)
    geo_matrix_v=torch.cuda.FloatTensor(geo_matrix_v)
    return torch.sparse.FloatTensor(geo_matrix_i.t(), geo_matrix_v)

sparse_u_u_geo=readTrainSparseMatrix_geo(table_geo_u2u,table_geo_distance_user,True)
sparse_i_i_geo=readTrainSparseMatrix_geo(table_geo_i2i,table_geo_distance_item,False)
# pdb.set_trace()

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num,user_item_matrix,item_user_matrix,user_user_matrix_geo,item_item_matrix_geo,d_i_train,d_j_train):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.user_user_matrix_geo=user_user_matrix_geo
        self.item_item_matrix_geo=item_item_matrix_geo
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 
        self.active=nn.LeakyReLU()

        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]]

        self.d_i_train=torch.cuda.FloatTensor(d_i_train)
        self.d_j_train=torch.cuda.FloatTensor(d_j_train)
        self.d_i_train=self.d_i_train.expand(-1,factor_num)
        self.d_j_train=self.d_j_train.expand(-1,factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)  

    def forward(self, user, item_i, item_j):    

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight  
        
        gcn1_users_embedding_u = (torch.sparse.mm(self.user_user_matrix_geo, users_embedding))
        gcn1_users_embedding_i = (torch.sparse.mm(self.user_item_matrix, items_embedding))
        gcn1_users_embedding = 0.5*gcn1_users_embedding_u+gcn1_users_embedding_i
        gcn1_items_embedding_u = (torch.sparse.mm(self.item_user_matrix, users_embedding))
        gcn1_items_embedding_i = (torch.sparse.mm(self.item_item_matrix_geo, items_embedding))
        gcn1_items_embedding = gcn1_items_embedding_u+0.5*gcn1_items_embedding_i

        
        gcn2_users_embedding_u = (torch.sparse.mm(self.user_user_matrix_geo, gcn1_users_embedding))
        gcn2_users_embedding_i = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding))
        gcn2_users_embedding = 0.5*gcn2_users_embedding_u+gcn2_users_embedding_i
        gcn2_items_embedding_u = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding))
        gcn2_items_embedding_i = (torch.sparse.mm(self.item_item_matrix_geo, gcn1_items_embedding))
        gcn2_items_embedding = gcn2_items_embedding_u+0.5*gcn2_items_embedding_i
        
        
        gcn3_users_embedding_u = (torch.sparse.mm(self.user_user_matrix_geo, gcn2_users_embedding))
        gcn3_users_embedding_i = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding))
        gcn3_users_embedding = 0.5*gcn3_users_embedding_u+gcn3_users_embedding_i
        gcn3_items_embedding_u = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding))
        gcn3_items_embedding_i = (torch.sparse.mm(self.item_item_matrix_geo, gcn2_items_embedding))
        gcn3_items_embedding = gcn3_items_embedding_u+0.5*gcn3_items_embedding_i
        
        # gcn4_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        # gcn4_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
        
        gcn_users_embedding= users_embedding+1/2*gcn1_users_embedding+1/3*gcn2_users_embedding+1/4*gcn3_users_embedding#+gcn4_users_embedding 
        gcn_items_embedding= items_embedding+1/2*gcn1_items_embedding+1/3*gcn2_items_embedding+1/4*gcn3_items_embedding#+gcn4_items_embedding 
      
        
        user = F.embedding(user,gcn_users_embedding)
        item_i = F.embedding(item_i,gcn_items_embedding)
        item_j = F.embedding(item_j,gcn_items_embedding)  
        #print(user.size())
        # # pdb.set_trace() 
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)#
        # loss=-((rediction_i-prediction_j).sigmoid())**2#self.loss(prediction_i,prediction_j)#.sum()
        l2_regulization = 0.0001*(user**2+item_i**2+item_j**2).sum(dim=-1)
        # l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())
      
        loss2= -((prediction_i - prediction_j).sigmoid().log().mean())
        # loss= loss2 + l2_regulization
        loss= -((prediction_i - prediction_j)).sigmoid().log().mean() +l2_regulization.mean()
        # pdb.set_trace()
        return prediction_i, prediction_j,loss,loss2
 
train_dataset = data_utils.BPRData(
        train_dict=training_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=training_set_count,all_rating=user_rating_set_all)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=0)#num_workers setted as 0
  
testing_dataset_loss = data_utils.BPRData(
        train_dict=testing_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=testing_set_count,all_rating=user_rating_set_all)
testing_loader_loss = DataLoader(testing_dataset_loss,
        batch_size=batch_size, shuffle=False, num_workers=0)

val_dataset_loss = data_utils.BPRData(
        train_dict=val_user_set, num_item=item_num, num_ng=###, is_training=True,\
        data_set_count=val_set_count,all_rating=user_rating_set_all)
val_loader_loss = DataLoader(val_dataset_loss,
        batch_size=batch_size, shuffle=False, num_workers=0)
   
   
model = BPR(user_num, item_num, factor_num,sparse_u_i,sparse_i_u,sparse_u_u_geo,sparse_i_i_geo,d_i_train,d_j_train)
model=model.to('cuda') 
#print(model.parameters().size())

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.01)#, betas=(0.5, 0.99))

########################### TRAINING #####################################
 
# testing_loader_loss.dataset.ng_sample() 

print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(1000):
    model.train() 
    start_time = time.time()
    train_loader.dataset.ng_sample()
    # pdb.set_trace()
    print('train data of ng_sample is  end')
    # elapsed_time = time.time() - start_time
    # print(' time:'+str(round(elapsed_time,1)))
    # start_time = time.time()
    
    train_loss_sum=[]
    train_loss_sum2=[]
    for user, item_i, item_j in train_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda() 

        model.zero_grad()
        prediction_i, prediction_j,loss,loss2 = model(user, item_i, item_j) 
        loss.backward()
        optimizer_bpr.step() 
        count += 1  
        train_loss_sum.append(loss.item())  
        train_loss_sum2.append(loss2.item())  
        # print(count)

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)
    train_loss2=round(np.mean(train_loss_sum2[:-1]),4)
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)+"="+str(train_loss2)+"+" 
    print('--train--',elapsed_time)

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    torch.save(model.state_dict(), PATH_model)
    

 

 


