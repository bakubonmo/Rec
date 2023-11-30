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
import math

dataset_base_path='../data/New_York'  
 

user_num=3286
item_num=6369 
factor_num=64
batch_size=2048*128
num_negative_test_val=-1
num_ng = ###
start_i_test=###
end_i_test=###
setp = ###
top_k = 5

run_id="s0"
print(run_id)
dataset='New_York'
path_save_base='./log/'+dataset+'/newloss'+run_id
if (os.path.exists(path_save_base)):
    print('has results save path')
else:
    os.makedirs(path_save_base)   
result_file=open(path_save_base+'/results.txt','w+')#('./log/results_gcmc.txt','w+')
copyfile('./train_New_York.py', path_save_base+'/train_New_York'+run_id+'.py')

path_save_model_base='../newlossModel/'+dataset+'/s'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)  



training_user_set_morn, training_item_set_morn,training_set_count_morn= np.load(dataset_base_path+'/datanpy/training_morn.npy',allow_pickle=True)
training_user_set_noon, training_item_set_noon,training_set_count_noon= np.load(dataset_base_path+'/datanpy/training_noon.npy',allow_pickle=True)
training_user_set_night, training_item_set_night,training_set_count_night= np.load(dataset_base_path+'/datanpy/training_night.npy',allow_pickle=True)
training_user_set_midnight, training_item_set_midnight,training_set_count_midnight= np.load(dataset_base_path+'/datanpy/training_midnight.npy',allow_pickle=True)


[table_geo_distance_user_item_general]=np.load(dataset_base_path+'/datanpy/table_geo_distance_user_item_general.npy',allow_pickle=True)
[table_geo_distance_item_user_general]=np.load(dataset_base_path+'/datanpy/table_geo_distance_item_user_general.npy',allow_pickle=True)

training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/datanpy/training_set.npy',allow_pickle=True)
testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/datanpy/testing_set.npy',allow_pickle=True)  
val_user_set,val_item_set,val_set_count = np.load(dataset_base_path+'/datanpy/val_set.npy',allow_pickle=True)    
user_rating_set_all = np.load(dataset_base_path+'/datanpy/user_rating_set_all.npy',allow_pickle=True).item()



def set_reciprocal(table_geo_distance):
    idx_nonzeros=np.where(table_geo_distance!=0)
    table_geo_distance[idx_nonzeros]=1/table_geo_distance[idx_nonzeros]
    maxi=table_geo_distance.max(axis=1)
    idx_nonzeros=np.where(maxi!=0)
    
    for i in idx_nonzeros[0]:
        table_geo_distance[i]=(table_geo_distance[i])/(maxi[i])
    
    return table_geo_distance


table_geo_distance_user_item_general=set_reciprocal(table_geo_distance_user_item_general)
table_geo_distance_item_user_general=set_reciprocal(table_geo_distance_item_user_general)
    
print("reciprocal end")
def readD(set_matrix,num_):
    user_d=[] 
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d
u_d_general=readD(training_user_set,user_num)
i_d_general=readD(training_item_set,item_num)
d_i_train=u_d_general
d_j_train=i_d_general


#user-item  to user-item matrix and item-user matrix
def readTrainSparseMatrix(set_matrix,u_d,i_d,is_user):
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
            if(len_set==0):
                d_i_j=0
            else:
                d_i_j=np.sqrt(d_i[i]*d_j[j])           
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

sparse_u_i=readTrainSparseMatrix(training_user_set,u_d_general,i_d_general,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,u_d_general,i_d_general,False)

def readTrainSparseMatrix_geo(set_matrix,u_d,i_d,geo_distance,is_user):
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
            if(len_set==0):
                d_i_j=0
            else:
                if(geo_distance[i][j]==0):
                    d_i_j=0
                else:
                    d_i_j=0.25*np.sqrt(d_i[i]*d_j[j])+0.75*np.sqrt(d_i[i])*(geo_distance[i][j])  
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

sparse_u_i_morn=readTrainSparseMatrix_geo(training_user_set_morn,u_d_general,i_d_general,table_geo_distance_user_item_general,True)
sparse_i_u_morn=readTrainSparseMatrix_geo(training_item_set_morn,u_d_general,i_d_general,table_geo_distance_item_user_general,False)
sparse_u_i_noon=readTrainSparseMatrix_geo(training_user_set_noon,u_d_general,i_d_general,table_geo_distance_user_item_general,True)
sparse_i_u_noon=readTrainSparseMatrix_geo(training_item_set_noon,u_d_general,i_d_general,table_geo_distance_item_user_general,False)
sparse_u_i_night=readTrainSparseMatrix_geo(training_user_set_night,u_d_general,i_d_general,table_geo_distance_user_item_general,True)
sparse_i_u_night=readTrainSparseMatrix_geo(training_item_set_night,u_d_general,i_d_general,table_geo_distance_item_user_general,False)
sparse_u_i_midnight=readTrainSparseMatrix_geo(training_user_set_midnight,u_d_general,i_d_general,table_geo_distance_user_item_general,True)
sparse_i_u_midnight=readTrainSparseMatrix_geo(training_item_set_midnight,u_d_general,i_d_general,table_geo_distance_item_user_general,False)

def save_metric_each_person(result,metric):
    save_path='./'+metric+'.csv'
    final_result=pd.DataFrame(data=result)
    final_result.to_csv(save_path, index = False, header=False,encoding='utf-8', mode='a')
    return

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num,user_item_matrix_morn,item_user_matrix_morn,user_item_matrix_noon,item_user_matrix_noon,user_item_matrix_night,item_user_matrix_night,user_item_matrix_midnight,item_user_matrix_midnight,user_item_matrix,item_user_matrix,d_i_train,d_j_train,training_user_set_morn, training_user_set_noon, training_user_set_night, training_user_set_midnight,table_geo_user_item):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.user_item_matrix_morn = user_item_matrix_morn
        self.item_user_matrix_morn = item_user_matrix_morn
        self.user_item_matrix_noon = user_item_matrix_noon
        self.item_user_matrix_noon = item_user_matrix_noon
        self.user_item_matrix_night = user_item_matrix_night
        self.item_user_matrix_night = item_user_matrix_night
        self.user_item_matrix_midnight = user_item_matrix_midnight
        self.item_user_matrix_midnight = item_user_matrix_midnight    
        
        self.user_item_matrix_general = user_item_matrix
        self.item_user_matrix_general = item_user_matrix
        self.training_user_set_morn = training_user_set_morn
        self.training_user_set_noon = training_user_set_noon
        self.training_user_set_night = training_user_set_night
        self.training_user_set_midnight = training_user_set_midnight
        self.d_i_train = d_i_train
        self.d_j_train = d_j_train
        self.table_geo_user_item = table_geo_user_item

        
       
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 
        self.weight_morn = nn.Parameter(torch.FloatTensor(factor_num))
        self.weight_noon = nn.Parameter(torch.FloatTensor(factor_num))
        self.weight_night = nn.Parameter(torch.FloatTensor(factor_num))
        self.weight_midnight = nn.Parameter(torch.FloatTensor(factor_num))
        self.active=nn.Sigmoid()
        
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01) 
        nn.init.normal_(self.weight_morn, mean=0, std=1)
        nn.init.normal_(self.weight_noon, mean=0, std=1)
        nn.init.normal_(self.weight_night, mean=0, std=1)
        nn.init.normal_(self.weight_midnight, mean=0, std=1)
        
        self.attn1 = nn.Linear(in_features=factor_num, out_features=factor_num,bias=True)
        self.attn2 = nn.Linear(in_features=factor_num, out_features=1,bias=False)
         
    
    def time_embedding(self,u_i_matrix, i_embedding, i_u_matrix, u_embedding, weight_matrix):
        gcn1_u_embedding_temp=(torch.sparse.mm(u_i_matrix, i_embedding))
        gcn1_u_embedding=(torch.mm(gcn1_u_embedding_temp, weight_matrix))
        gcn1_i_embedding_temp=(torch.sparse.mm(i_u_matrix, u_embedding))
        gcn1_i_embedding=(torch.mm(gcn1_i_embedding_temp, weight_matrix))
        
        gcn2_u_embedding_temp=(torch.sparse.mm(u_i_matrix, gcn1_i_embedding))
        gcn2_u_embedding=(torch.mm(gcn2_u_embedding_temp, weight_matrix))
        gcn2_i_embedding_temp=(torch.sparse.mm(i_u_matrix, gcn1_u_embedding))
        gcn2_i_embedding=(torch.mm(gcn2_i_embedding_temp, weight_matrix))
        
        gcn3_u_embedding_temp=(torch.sparse.mm(u_i_matrix, gcn2_i_embedding))
        gcn3_u_embedding=(torch.mm(gcn3_u_embedding_temp, weight_matrix))
        gcn3_i_embedding_temp=(torch.sparse.mm(i_u_matrix, gcn2_u_embedding))
        gcn3_i_embedding=(torch.mm(gcn3_i_embedding_temp, weight_matrix))
        
        gcn_u_embedding=1/2*gcn1_u_embedding+1/3*gcn2_u_embedding+1/4*gcn3_u_embedding
        gcn_i_embedding=1/2*gcn1_i_embedding+1/3*gcn2_i_embedding+1/4*gcn3_i_embedding
        

        return gcn_u_embedding, gcn_i_embedding
    
    def attn_weight(self, embedding_morn, embedding_noon, embedding_night, embedding_midnight):
        w_morn=self.attn2(torch.tanh(self.attn1(embedding_morn)))
        w_noon=self.attn2(torch.tanh(self.attn1(embedding_noon)))
        w_night=self.attn2(torch.tanh(self.attn1(embedding_night)))
        w_midnight=self.attn2(torch.tanh(self.attn1(embedding_midnight)))
        
        alpha=torch.cat((w_morn, w_noon, w_night, w_midnight),dim=1)
        alpha=F.normalize(alpha,p=2, dim=1)

        alpha_morn=alpha[:,0].view(len(alpha),1)
        alpha_noon=alpha[:,1].view(len(alpha),1)
        alpha_night=alpha[:,2].view(len(alpha),1)
        alpha_midnight=alpha[:,3].view(len(alpha),1)
        return alpha_morn, alpha_noon, alpha_night, alpha_midnight
        
    
    def bool_matrix(self, pred_cluster_morn, pred_cluster_noon, pred_cluster_night, pred_cluster_midnight, user, item_i):  
        
        training_user_set_morn=defaultdict(set)
        training_user_set_noon=defaultdict(set)
        training_user_set_night=defaultdict(set)
        training_user_set_midnight=defaultdict(set)
          
        average_morn = pred_cluster_morn.mean()
        average_noon = pred_cluster_noon.mean()
        average_night = pred_cluster_night.mean()
        average_midnight = pred_cluster_midnight.mean()
        
     
        for i in range(int(len(pred_cluster_morn)/(num_ng*75))):
            average = (pred_cluster_morn[i]+pred_cluster_noon[i]+pred_cluster_night[i]+pred_cluster_midnight[i])/4
            if(pred_cluster_morn[i]>average):
                if(item_i[i] not in training_user_set_morn[user[i]]):
                    training_user_set_morn[user[i]].add(item_i[i])
            else:
                training_user_set_morn[user[i]].discard(item_i[i])
           
            if(pred_cluster_noon[i]>average):
                if(item_i[i] not in training_user_set_noon[user[i]]):
                    training_user_set_noon[user[i]].add(item_i[i])
            else:
                training_user_set_noon[user[i]].discard(item_i[i])
            if(pred_cluster_night[i]>average):
                if(item_i[i] not in training_user_set_night[user[i]]):
                    training_user_set_night[user[i]].add(item_i[i])
            else:
                training_user_set_night[user[i]].discard(item_i[i])
            if(pred_cluster_midnight[i]>average):
                if(item_i[i] not in training_user_set_midnight[user[i]]):
                    training_user_set_midnight[user[i]].add(item_i[i])
            else:
                training_user_set_midnight[user[i]].discard(item_i[i])
        
        
        self.user_item_matrix_morn = self.readTrainSparseMatrix(training_user_set_morn,self.d_i_train,self.d_j_train,self.table_geo_user_item,True)
        self.user_item_matrix_noon = self.readTrainSparseMatrix(training_user_set_noon,self.d_i_train,self.d_j_train,self.table_geo_user_item,True)
        self.user_item_matrix_night = self.readTrainSparseMatrix(training_user_set_night,self.d_i_train,self.d_j_train,self.table_geo_user_item,True)
        self.user_item_matrix_midnight = self.readTrainSparseMatrix(training_user_set_midnight,self.d_i_train,self.d_j_train,self.table_geo_user_item,True)
        
        self.item_user_matrix_morn = self.user_item_matrix_morn.t()
        self.item_user_matrix_noon = self.user_item_matrix_noon.t()
        self.item_user_matrix_night = self.user_item_matrix_night.t()
        self.item_user_matrix_midnight = self.user_item_matrix_midnight.t()
        return
        
    def readD(set_matrix):
        user_d=[] 
        for i in range(len(set_matrix)):
            len_set=1.0/(len(set_matrix[i])+1)  
            user_d.append(len_set)
        return user_d
    
    def readTrainSparseMatrix(self,set_matrix,u_d,i_d,geo_distance,is_user):
        
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
                if(len_set==0):
                    d_i_j=0
                else:
                    d_i_j=0.25*np.sqrt(d_i[i]*d_j[j])+0.75*np.sqrt(d_i[i])*(geo_distance[i][j])     
                user_items_matrix_v.append(d_i_j)
        user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
        user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
        
        return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)
    

    def forward(self, user, item_i, item_j):

        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight  
        
        weight_matrix_morn=torch.diag(self.weight_morn)
        weight_matrix_noon=torch.diag(self.weight_noon)
        weight_matrix_night=torch.diag(self.weight_night)
        weight_matrix_midnight=torch.diag(self.weight_midnight)
        
        gcn_users_embedding_morn,gcn_items_embedding_morn=self.time_embedding(self.user_item_matrix_morn, items_embedding, self.item_user_matrix_morn, users_embedding, weight_matrix_morn)
        gcn_users_embedding_noon,gcn_items_embedding_noon=self.time_embedding(self.user_item_matrix_noon, items_embedding, self.item_user_matrix_noon, users_embedding, weight_matrix_noon)
        gcn_users_embedding_night,gcn_items_embedding_night=self.time_embedding(self.user_item_matrix_night, items_embedding, self.item_user_matrix_night, users_embedding, weight_matrix_night)
        gcn_users_embedding_midnight,gcn_items_embedding_midnight=self.time_embedding(self.user_item_matrix_midnight, items_embedding, self.item_user_matrix_midnight, users_embedding, weight_matrix_midnight)
        
        
        alpha_user_morn, alpha_user_noon, alpha_user_night, alpha_user_midnight = self.attn_weight(gcn_users_embedding_morn, gcn_users_embedding_noon, gcn_users_embedding_night, gcn_users_embedding_midnight)
        alpha_item_morn, alpha_item_noon, alpha_item_night, alpha_item_midnight = self.attn_weight(gcn_items_embedding_morn, gcn_items_embedding_noon, gcn_items_embedding_night, gcn_items_embedding_midnight)
        
        
        gcn_users_embedding_time= alpha_user_morn*gcn_users_embedding_morn + alpha_user_noon*gcn_users_embedding_noon + alpha_user_night*gcn_users_embedding_night + alpha_user_midnight*gcn_users_embedding_midnight
        gcn_items_embedding_time= alpha_item_morn*gcn_items_embedding_morn + alpha_item_noon*gcn_items_embedding_noon + alpha_item_night*gcn_items_embedding_night + alpha_item_midnight*gcn_items_embedding_midnight
        
        
        gcn_users_embedding= users_embedding+gcn_users_embedding_time
        gcn_items_embedding= items_embedding+gcn_items_embedding_time
      
        
        users = F.embedding(user,gcn_users_embedding)
        items_i = F.embedding(item_i,gcn_items_embedding)
        items_j = F.embedding(item_j,gcn_items_embedding)  
        
        item_0 = F.embedding(item_i,items_embedding)
        user_morn = F.embedding(user,gcn_users_embedding_morn)
        user_noon = F.embedding(user,gcn_users_embedding_noon)
        user_night = F.embedding(user,gcn_users_embedding_night)
        user_midnight = F.embedding(user,gcn_users_embedding_midnight)
        
        
        pred_cluster_morn = (user_morn * item_0).sum(dim=-1).cpu()
        pred_cluster_noon = (user_noon * item_0).sum(dim=-1).cpu()
        pred_cluster_night = (user_night * item_0).sum(dim=-1).cpu()
        pred_cluster_midnight = (user_midnight * item_0).sum(dim=-1).cpu()
        
        print(pred_cluster_morn.mean(),pred_cluster_noon.mean(), pred_cluster_night.mean(),pred_cluster_midnight.mean())
        user.cpu()
        item_i.cpu()
        self.bool_matrix(pred_cluster_morn, pred_cluster_noon, pred_cluster_night, pred_cluster_midnight,user, item_i)     
      
        
        g0_mean=torch.mean(users_embedding)
        g0_var=torch.var(users_embedding)
        g_mean=torch.mean(gcn_users_embedding)
        g_var=torch.var(gcn_users_embedding)

        i0_mean=torch.mean(items_embedding)
        i0_var=torch.var(items_embedding)
        i_mean=torch.mean(gcn_items_embedding)
        i_var=torch.var(gcn_items_embedding)

        # pdb.set_trace() 

        str_user=str(round(g0_mean.item(),7))+' '
        str_user+=str(round(g0_var.item(),7))+' '
        str_user+=str(round(g_mean.item(),7))+' '
        str_user+=str(round(g_var.item(),7))+' '

        str_item=str(round(i0_mean.item(),7))+' '
        str_item+=str(round(i0_var.item(),7))+' '
        str_item+=str(round(i_mean.item(),7))+' '
        str_item+=str(round(i_var.item(),7))+' '

        #print(str_user)
        #print(str_item)


        return gcn_users_embedding, gcn_items_embedding,str_user,str_item 


 
 
test_batch=52
testing_dataset = data_utils.resData(train_dict=testing_user_set, batch_size=test_batch,num_item=item_num,all_pos=training_user_set)
testing_loader = DataLoader(testing_dataset,batch_size=1, shuffle=False, num_workers=0) 
 

model = BPR(user_num, item_num, factor_num,sparse_u_i_morn,sparse_i_u_morn,sparse_u_i_noon,sparse_i_u_noon,sparse_u_i_night,sparse_i_u_night,sparse_u_i_midnight,sparse_i_u_midnight,sparse_u_i,sparse_i_u,d_i_train,d_j_train,training_user_set_morn, training_user_set_noon, training_user_set_night, training_user_set_midnight,table_geo_distance_user_item_general)
model=model.to('cuda')
   
optimizer_bpr = torch.optim.Adam(model.parameters(), lr=####)

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
        
        recall_t,ndcg_t,precision=evaluate.recall_ndcg(indices,index_end_i,top_k)
        elapsed_time = time.time() - test_start_time 
        Recall.append(recall_t)
        NDCG.append(ndcg_t)  
        PREC.append(precision)
        
    recall_test=round(np.mean(Recall),4)
    ndcg_test=round(np.mean(NDCG),4)  
    prec_test=round(np.mean(PREC),4)
    
    str_print_evl="epoch:"+str(epoch)+'time:'+str(round(elapsed_time,2))+"\t test"+" Recall:"+str(recall_test)+' ndcg:'+str(ndcg_test)+' prec:'+str(prec_test)
    print(str_print_evl)   
    result_file.write(gcn_user_emb)
    result_file.write('\n')
    result_file.write(gcn_item_emb)
    result_file.write('\n')  

    result_file.write(str_print_evl)
    result_file.write('\n')
    result_file.flush()

 

 


