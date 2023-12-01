
import pdb
from collections import defaultdict
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time

training_path='./train.txt' 
coos_path='./coos.txt'

user_num=18737
item_num=32510


def cal_geo_distance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) 
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 
    distance=round(distance/1000,3)
    return distance

def cal_table_geo_distance(item_geo_inform):
    table_geo_distance=np.zeros([item_num,item_num])
    for i in range(item_num):
        #rint(i)
        for j in range(i+1,item_num):
            lng1=item_geo_inform[i][1]
            lat1=item_geo_inform[i][0]
            lng2=item_geo_inform[j][1]
            lat2=item_geo_inform[j][0]
            distance=cal_geo_distance(lng1,lat1,lng2,lat2)
            table_geo_distance[i][j]=distance
            table_geo_distance[j][i]=distance
    #np.save('./datanpy/training_geo_set.npy',[table_geo_distance])
    return table_geo_distance


def cal_geo_neighbor(table_geo_distance, lambda_distance):
    train_geo_i2i = defaultdict(set)
    for p_id in range(item_num):     
        geo_neighbor=table_geo_distance[p_id]
        geo_neighbor=np.where((geo_neighbor<lambda_distance) & (geo_neighbor>=0))
        #geo_neighbor=np.where(geo_neighbor[0]>0)
        #print(geo_neighbor,len(geo_neighbor[0]))
        for i in geo_neighbor[0]:
            train_geo_i2i[p_id].add(i)
            #train_geo_i2i[p_id].add((i,table_geo_distance[p_id][i]))
        for i in np.where(table_geo_distance[p_id]>=lambda_distance):
            table_geo_distance[p_id][i]=0
    #np.save('./datanpy/training_geo_set.npy',[table_geo_distance])
    np.save('./datanpy/training_geo_neighbor_i2i.npy',[train_geo_i2i])
    return train_geo_i2i




def DBSCAN_cluster(train_data_user_geo):
    user_centers=[[] for i in range(user_num)]
    for i in range(user_num):
        #print(i)
        X = np.array(train_data_user_geo[i])
    
        kms_per_radian = 6371.0088
        epsilon = 1.0*1000 / kms_per_radian
        estimator = DBSCAN(eps = epsilon, min_samples = 2,metric = 'haversine').fit(np.radians(X))
        estimator.fit(X)
        label_pred = estimator.labels_ 
        #print(label_pred)
        cluster_num=max(label_pred)
        for j in range(cluster_num+1):
            indexs=[k for k in range(len(label_pred)) if label_pred[k]==j]
            lats=[X[k][0] for k in indexs]
            lngs=[X[k][1] for k in indexs]
            #print(lats)
            #print(lngs)
            avg_lat=sum(lats)/len(lats)
            avg_lng=sum(lngs)/len(lngs)
            #print(avg_lat,avg_lng)
            #print("----------------")
            user_centers[i].append([avg_lat,avg_lng])

    return user_centers

def cal_table_geo_distance_user(user_centers):
    table_geo_distance=np.zeros([user_num,user_num])
    for i in range(user_num):
        #print(i)
        centers_i=user_centers[i]
        for j in range(i+1,user_num):
            centers_j=user_centers[j]
            min_distance=1000000
            for [lat1,lng1] in centers_i:
                for [lat2, lng2] in centers_j:
                    distance=cal_geo_distance(lng1,lat1,lng2,lat2)
                    if distance<min_distance:
                        min_distance=distance
            table_geo_distance[i][j]=min_distance
            table_geo_distance[j][i]=min_distance
    return table_geo_distance


def cal_geo_neighbor_user(table_geo_distance_user, lambda_distance):
    train_geo_u2u = defaultdict(set)
    for u_id in range(user_num):
        geo_neighbor=table_geo_distance_user[u_id]
        geo_neighbor=np.where((geo_neighbor<lambda_distance) & (geo_neighbor>=0))
        for i in geo_neighbor[0]:
            train_geo_u2u[u_id].add(i)
            #train_geo_u2u[u_id].add((i,table_geo_distance_user[u_id][i]))
         
        #for i in np.where(table_geo_distance_user[u_id]>=lambda_distance):
            #table_geo_distance_user[u_id][i]=0    
    #np.save('./datanpy/training_geo_user_set.npy',[table_geo_distance_user])
    np.save('./datanpy/training_geo_neighbor_u2u.npy',[train_geo_u2u])
    return train_geo_u2u

def filter_larger(table_geo_distance,num, lambda_distance):
    for i in range(num):
        filter_neighbor=table_geo_distance[i]
        filter_neighbor=np.where(filter_neighbor>=lambda_distance)
        for j in filter_neighbor[0]:
            table_geo_distance[i][j]=0
    return table_geo_distance
    
if __name__ == '__main__':
    path_save_base='./datanpy'
    if (os.path.exists(path_save_base)):
        print('has geo results save path')
    else:
        os.makedirs(path_save_base)  
    
    f_geo=open(coos_path)
    item_geo_inform=np.zeros([item_num,2])
    #table_geo_distance=defaultdict(lambda:dict())
    for line in f_geo.readlines():
        p_id,p_lat,p_lng=line.split()
        item_geo_inform[int(p_id)][0]=float(p_lat)
        item_geo_inform[int(p_id)][1]=float(p_lng)
    
    start1=time.clock()
    table_geo_distance=cal_table_geo_distance(item_geo_inform)
    #np.save('./datanpy/training_geo_set.npy',[table_geo_distance])
    end1=time.clock()
    #[table_geo_distance] = np.load('./datanpy/training_geo_set.npy',allow_pickle=True)
    train_geo_i2i=cal_geo_neighbor(table_geo_distance, 0.75)
    #np.save('./datanpy/training_geo_set.npy',[table_geo_distance])
    start2=time.clock()
    ##########user
    f_train=open(training_path)
    train_data_user = defaultdict(set)
    train_data_user_geo=[[] for i in range(user_num)]
    for line in f_train.readlines():
        u_ps=line.split()
        u_id=int(u_ps[0])
        for p_id in u_ps[1:]:
            train_data_user[u_id].add(int(p_id))
            train_data_user_geo[u_id].append([item_geo_inform[int(p_id)][0],item_geo_inform[int(p_id)][1]])
    
    
    user_centers=DBSCAN_cluster(train_data_user_geo)
    table_geo_distance_user=cal_table_geo_distance_user(user_centers)
    #np.save('./datanpy/training_geo_user_set.npy',[table_geo_distance_user])
    
    end2=time.clock()
    #[table_geo_distance_user] = np.load('./datanpy/training_geo_user_set.npy',allow_pickle=True)
    table_geo_u2u=cal_geo_neighbor_user(table_geo_distance_user, 0.75)
    #np.save('./datanpy/training_geo_user_set.npy',[table_geo_distance_user])
    
    
    table_geo_distance_user=filter_larger(table_geo_distance_user,user_num, 0.75)
    np.save('./datanpy/training_geo_user_set.npy',[table_geo_distance_user])
    table_geo_distance=filter_larger(table_geo_distance,item_num, 0.75)
    np.save('./datanpy/training_geo_set.npy',[table_geo_distance])
    
    print((end1-start1)+(end2-start2))
    