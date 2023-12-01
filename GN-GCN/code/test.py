# -- coding:UTF-8 
import torch
# print(torch.__version__) 
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [1]))

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
import pandas as pd

dataset_base_path='../data/gowalla-geo'  
 
##gowalla
user_num=18737
item_num=32510 
factor_num=64
batch_size=2048*512
top_k=5
num_negative_test_val=-1##all   

start_i_test=###
end_i_test=###
setp=1
 

run_id="s0"
print(run_id)
dataset='gowalla-geo'
path_save_base='./log/'+dataset+'/newloss'+run_id
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    print('error') 
    pdb.set_trace() 
result_file=open(path_save_base+'/results_hdcg_hr.txt','a')#('./log/results_gcmc.txt','w+')
copyfile('./test_gowalla_geo.py', path_save_base+'/test_gowalla_geo'+run_id+'.py')

path_save_model_base='../newlossModel/'+dataset+'/s'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    pdb.set_trace() 
 

   
training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/datanpy/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/datanpy/testing_set.npy',allow_pickle=True)   
user_rating_set_all = np.load(dataset_base_path+'/datanpy/user_rating_set_all.npy',allow_pickle=True).item()
[table_geo_u2u] = np.load(dataset_base_path+'/datanpy/training_geo_neighbor_u2u.npy',allow_pickle=True)#
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
# pdb.set_trace()

def save_metric_each_person(result,metric):
    save_path='./'+metric+'.csv'
    final_result=pd.DataFrame(data=result)
    final_result.to_csv(save_path, index = False, header=False,encoding='utf-8', mode='a')
    return

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
        gcn1_users_embedding = 0.5*self.active(gcn1_users_embedding_u)+gcn1_users_embedding_i
        gcn1_items_embedding_u = (torch.sparse.mm(self.item_user_matrix, users_embedding))
        gcn1_items_embedding_i = (torch.sparse.mm(self.item_item_matrix_geo, items_embedding))
        gcn1_items_embedding = gcn1_items_embedding_u+0.5*self.active(gcn1_items_embedding_i)

        
        gcn2_users_embedding_u = (torch.sparse.mm(self.user_user_matrix_geo, gcn1_users_embedding))
        gcn2_users_embedding_i = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding))
        gcn2_users_embedding = 0.5*self.active(gcn2_users_embedding_u)+gcn2_users_embedding_i
        gcn2_items_embedding_u = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding))
        gcn2_items_embedding_i = (torch.sparse.mm(self.item_item_matrix_geo, gcn1_items_embedding))
        gcn2_items_embedding = gcn2_items_embedding_u+0.5*self.active(gcn2_items_embedding_i)
        
        
        gcn3_users_embedding_u = (torch.sparse.mm(self.user_user_matrix_geo, gcn2_users_embedding))
        gcn3_users_embedding_i = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding))
        gcn3_users_embedding = 0.5*self.active(gcn3_users_embedding_u)+gcn3_users_embedding_i
        gcn3_items_embedding_u = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding))
        gcn3_items_embedding_i = (torch.sparse.mm(self.item_item_matrix_geo, gcn2_items_embedding))
        gcn3_items_embedding = gcn3_items_embedding_u+0.5*self.active(gcn3_items_embedding_i)
       
        gcn_users_embedding= users_embedding+1/2*gcn1_users_embedding+1/3*gcn2_users_embedding+1/4*gcn3_users_embedding
        gcn_items_embedding= items_embedding+1/2*gcn1_items_embedding+1/3*gcn2_items_embedding+1/4*gcn3_items_embedding
        
        
        g0_mean=torch.mean(users_embedding)
        g0_var=torch.var(users_embedding)
        g1_mean=torch.mean(gcn1_users_embedding)
        g1_var=torch.var(gcn1_users_embedding) 
        g2_mean=torch.mean(gcn2_users_embedding)
        g2_var=torch.var(gcn2_users_embedding)
        g3_mean=torch.mean(gcn3_users_embedding)
        g3_var=torch.var(gcn3_users_embedding)
        # g4_mean=torch.mean(gcn4_users_embedding)
        # g4_var=torch.var(gcn4_users_embedding)
        # g5_mean=torch.mean(gcn5_users_embedding)
        # g5_var=torch.var(gcn5_users_embedding)
        # g6_mean=torch.mean(gcn6_users_embedding)
        # g6_var=torch.var(gcn6_users_embedding)
        g_mean=torch.mean(gcn_users_embedding)
        g_var=torch.var(gcn_users_embedding)

        i0_mean=torch.mean(items_embedding)
        i0_var=torch.var(items_embedding)
        i1_mean=torch.mean(gcn1_items_embedding)
        i1_var=torch.var(gcn1_items_embedding)
        i2_mean=torch.mean(gcn2_items_embedding)
        i2_var=torch.var(gcn2_items_embedding)
        i3_mean=torch.mean(gcn3_items_embedding)
        i3_var=torch.var(gcn3_items_embedding)
        # i4_mean=torch.mean(gcn4_items_embedding)
        # i4_var=torch.var(gcn4_items_embedding) 
        # i5_mean=torch.mean(gcn5_items_embedding)
        # i5_var=torch.var(gcn5_items_embedding)
        # i6_mean=torch.mean(gcn6_items_embedding)
        # i6_var=torch.var(gcn6_items_embedding)
        i_mean=torch.mean(gcn_items_embedding)
        i_var=torch.var(gcn_items_embedding)

        # pdb.set_trace() 

        str_user=str(round(g0_mean.item(),7))+' '
        str_user+=str(round(g0_var.item(),7))+' '
        str_user+=str(round(g1_mean.item(),7))+' '
        str_user+=str(round(g1_var.item(),7))+' '
        str_user+=str(round(g2_mean.item(),7))+' '
        str_user+=str(round(g2_var.item(),7))+' '
        str_user+=str(round(g3_mean.item(),7))+' '
        str_user+=str(round(g3_var.item(),7))+' '
        # str_user+=str(round(g4_mean.item(),7))+' '
        # str_user+=str(round(g4_var.item(),7))+' '
        # str_user+=str(round(g5_mean.item(),7))+' '
        # str_user+=str(round(g5_var.item(),7))+' '
        # str_user+=str(round(g6_mean.item(),7))+' '
        # str_user+=str(round(g6_var.item(),7))+' '
        str_user+=str(round(g_mean.item(),7))+' '
        str_user+=str(round(g_var.item(),7))+' '

        str_item=str(round(i0_mean.item(),7))+' '
        str_item+=str(round(i0_var.item(),7))+' '
        str_item+=str(round(i1_mean.item(),7))+' '
        str_item+=str(round(i1_var.item(),7))+' '
        str_item+=str(round(i2_mean.item(),7))+' '
        str_item+=str(round(i2_var.item(),7))+' '
        str_item+=str(round(i3_mean.item(),7))+' '
        str_item+=str(round(i3_var.item(),7))+' '
        # str_item+=str(round(i4_mean.item(),7))+' '
        # str_item+=str(round(i4_var.item(),7))+' '
        # str_item+=str(round(i5_mean.item(),7))+' '
        # str_item+=str(round(i5_var.item(),7))+' '
        # str_item+=str(round(i6_mean.item(),7))+' '
        # str_item+=str(round(i6_var.item(),7))+' '
        str_item+=str(round(i_mean.item(),7))+' '
        str_item+=str(round(i_var.item(),7))+' '

        print(str_user)
        print(str_item)


        return gcn_users_embedding, gcn_items_embedding,str_user,str_item 


 
 
test_batch=52#int(batch_size/32) 
testing_dataset = data_utils.resData(train_dict=testing_user_set, batch_size=test_batch,num_item=item_num,all_pos=training_user_set)
testing_loader = DataLoader(testing_dataset,batch_size=1, shuffle=False, num_workers=0) 
 

model = BPR(user_num, item_num, factor_num,sparse_u_i,sparse_i_u,sparse_u_u_geo,sparse_i_i_geo,d_i_train,d_j_train)
model=model.to('cuda')
   
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.01)#, betas=(0.5, 0.99))

########################### TRAINING ##################################### 
# testing_loader_loss.dataset.ng_sample() 

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

print('--------test processing-------')
count, best_hr = 0, 0
for epoch in range(start_i_test,end_i_test,setp):
    model.train()   

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    #torch.save(model.state_dict(), PATH_model) 
    model.load_state_dict(torch.load(PATH_model)) 
    model.eval()     

    # ######test and val###########    
    gcn_users_embedding, gcn_items_embedding,gcn_user_emb,gcn_item_emb= model(torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0]), torch.cuda.LongTensor([0])) 
    user_e=gcn_users_embedding.cpu().detach().numpy()
    item_e=gcn_items_embedding.cpu().detach().numpy()
    all_pre=np.matmul(user_e,item_e.T) 
    Recall, NDCG ,PREC= [], [], []
    set_all=set(range(item_num))  
    test_start_time = time.time()
    for u_i in testing_user_set: 
       
        item_i_list = list(testing_user_set[u_i])
        index_end_i=len(item_i_list)
        item_j_list = list(set_all-training_user_set[u_i]-testing_user_set[u_i])
        item_i_list.extend(item_j_list) 
        pre_one=all_pre[u_i][item_i_list] 
        indices=largest_indices(pre_one, top_k)
        indices=list(indices[0]) 
        recall_t,ndcg_t,precision=evaluate.hr_ndcg(indices,index_end_i,top_k)
        elapsed_time = time.time() - test_start_time 
        Recall.append(hr_t)
        NDCG.append(ndcg_t)  
        PREC.append(precision)
      
    recall_test=round(np.mean(Recall),4)
    ndcg_test=round(np.mean(NDCG),4)  
    prec_test=round(np.mean(PREC),4)
    save_metric_each_person(Recall,"Recall")
    save_metric_each_person(NDCG,"NDCG")
    save_metric_each_person(PREC,"PREC")
 
    str_print_evl="epoch:"+str(epoch)+'time:'+str(round(elapsed_time,2))+"\t test"+" recall:"+str(recall_test)+' ndcg:'+str(ndcg_test)+' prec:'+str(prec_test)
    print(str_print_evl)   
    result_file.write(gcn_user_emb)
    result_file.write('\n')
    result_file.write(gcn_item_emb)
    result_file.write('\n')  

    result_file.write(str_print_evl)
    result_file.write('\n')
    result_file.flush()

 

 
 


