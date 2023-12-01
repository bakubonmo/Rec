
#This file is used for generate result of DA
#Run this file with ```python newdata_withgreedy_1124.py INPUTFOLDER OUTPUTFILENAME MINUTE```
#Argument 1: INPUT FOLDER is the path of input files, often named as datasetxx, containing probabilities (named as cat_prob_DATETIME)
#            and distributions (named category_distribution)
#Argument 2: OUTPUTFILENAME is the output log. The filename is as variable ```result_file```
#Argument 3: MINUTE is the minute, for example 60.

#Flow of program:
#  This program will first read the constraints. Then, for each time slot, it reads the probability and 
#  generate Ising Model. Then, send the Ising Model (matrics) to DA. After receiving result from DA,
#  greedy strategy will be executed to generate final result.

#Output:
#  All output is in the outputfile. The useful columns are 
#  - dausergreedy: Precision of DA result, **this is the base for other evaluation calculation**
#  - time: the time slot (e.g. 1 o'clock or 14 o'clock)
#  - amount: amount of users in corresponding time slot
#  - cnttime: time that DA spent for calculation
#  - dahit: violation_rate = 1 - dahit
import numpy as np


#   This function received file argument and number of cat argument. It will return 2 lists,
# first return value is the constraint ratio (e.g. 0.98 for a category and 0.02 for another category)
# second return value is the number(name/id) of category (e.g. category 23)
def read_constraints(file,num_cat=6):
    category_constraint_each_hour = []
    category_needed_each_hour = []
    header = file.readline()
    #the first column is date&hour
    header_arr = np.array([int(i) for i in header.split(',')[1:]])
    for line in file.readlines():
        #read the portion
        tmp = np.array([float(i) for i in line.split(',')[1:]])
        #sort the portion
        s = np.argsort(tmp)[::-1]
        category_needed_each_hour.append(header_arr[s][:num_cat])
        category_constraint_each_hour.append(tmp[s][:num_cat])
    return category_constraint_each_hour,category_needed_each_hour


# This function read probability files (by given file parameter), categories is also needed
# This function returns the probability and ground truth
def read_file(file,categories,num_cat=6):
    #print(categories)
    cat_dict = {}
    for i,cat in enumerate(categories):
        cat_dict[cat]=i
    #print(cat_dict)
    header = file.readline()
    #intialize header arr
    #all_header_arr = [int(i) for i in header.split(',')]#first column is gt, so ignored
    header_arr = [int(i) for i in header.split(',')[1:]]
    
    #put the index into header_index
    header_index = []
    for header_cat in (header_arr):
        for j,cat in enumerate(categories):
            if header_cat==cat:
                header_index.append(j)
                break
        #There is no identical category in categories
        else:
            header_index.append(-1)
    #print(header_index)
    #assert(len(header_index)==num_cat)
    file_values = []
    #put each line into the matrix
    for line in file.readlines():
        #tmp save each row
        tmp = np.zeros((num_cat+1,))
        values = line.split(',')
        #save the probs
        for i,value in enumerate(values[1:]):
            save_index = header_index[i]
            #the probability of this category is not needed (because the category is not included in this hour)
            if save_index==-1:
                continue
            tmp[save_index]=float(value)
        #save the gt
        #print(values)
        if values[0]=='[]':#empty gt
            gt = -2
            tmp[num_cat]=gt
        else:
            gt = values[0][1:-1].strip().split()[0]
            tmp[num_cat]=cat_dict.get(int(gt),-1)
        file_values.append(tmp)
    prob_and_gt = np.array(file_values)
    return prob_and_gt

#This function generate the matrics and vectors for DA
def generate_da_constraints(PORTION,N=4096,NUM_USER=470,NUM_CAT=6):
    st=time.time()
    assert(len(PORTION)==NUM_CAT)
    hori_mat = np.zeros((N,N),dtype=int)
    hori_vector = np.full((N,),2**30,dtype=int)
    vert_mat = np.zeros((N,N),dtype=int)
    vert_vector = np.full((N,),2**30,dtype=int)
    hori_const = NUM_USER
    vert_const = sum(i**2 for i in PORTION)
    for i in range(NUM_USER*NUM_CAT):
        hori_vector[i]=-1
        vert_vector[i]=1+-2*PORTION[i%NUM_CAT]
    from itertools import combinations
    for i in range(NUM_USER):
        for x,y in combinations(range(NUM_CAT),2):
            hori_mat[i*NUM_CAT+x][i*NUM_CAT+y] = hori_mat[i*NUM_CAT+y][i*NUM_CAT+x] = 2
    for i in range(NUM_CAT):
        for x,y in combinations(range(NUM_USER),2):
            vert_mat[x*NUM_CAT+i][y*NUM_CAT+i] = vert_mat[y*NUM_CAT+i][x*NUM_CAT+i] = 2
    print(time.time()-st,end='\t')
    return (hori_mat,vert_mat,hori_vector,vert_vector,hori_const,vert_const)


import time
import sys
from python_fjda import fjda
if sys.version_info[0] == 3:
    import lzma
    long = int
    

#  This function set the parameters that DA needs. This function is altered from official example,so please
#don't change it unless you need to change the parameter of DA itself.
def setup(**kwargs):
    for k in kwargs.keys():
        if k not in ['weight', 'bias', 'constant', 'state', 'num_run', 'num_iteration','num_bit']:
            raise KeyError('unknown argument: %s' % k)
    w = kwargs['weight']
    b = kwargs['bias']
    c = kwargs['constant']
    s = kwargs.get('state', np.zeros(w.shape[0], dtype=int))
    num_bit = kwargs.get('num_bit', 4096)
    num_run = kwargs.get('num_run', 128)
    num_iteration = kwargs.get('num_iteration', 10**5)
    if w.shape[0] != w.shape[1]:
        raise ValueError('weight is not square matrix')
    if w.shape[0] != b.shape[0]:
        raise ValueError('weight, bias: size mismatch')
    if w.shape[0] != s.shape[0]:
        raise ValueError('weight, state: size mismatch')
    if not np.all(np.logical_or(s==0, s==1)):
        raise ValueError('state values must be 0 or 1')
    local_field = np.dot(w, s) + b
    E0 = np.dot(np.dot(w,s), s)/(-2) - np.dot(b,s) + c
    str_state = ''.join([chr(ord('0')+i) for i in s.tolist()]) 
    args = {'weight': w.ravel().tolist(),
            'lf_i_n': local_field.tolist()*num_run,
             'state_min_i_n': str_state*num_run,
             'state_cur_i_n': str_state*num_run,
             # 'tmp_i_n': None,
             'eg_min_i_n': [long(E0)]*num_run,
             'eg_cur_i_n': [long(E0)]*num_run,
             'num_iteration': num_iteration,
             'num_bit': num_bit,
             'num_run': num_run,
             'ope_mode': 2}
    param = {'offset_mode': 3,
             'offset_inc_rate': 1000,
             'criterion_for_escape': 1,
             'tmp_st': 655.36,
             'tmp_decay': 0.0,
             'tmp_mode': 0,
             'tmp_interval': 100,
             #'user_tmp': None,
             #'user_iteration': None,
             'repex_mode': 4,
             'repex_interval': 100,
             'repex_group_num': 1}
    #print('args and params preparation finished',time.time())
    return args, param

#Start DA running. This function is also altered from officila example
def run(args, param, debug=False):
    st = time.time()
    da = fjda.fjda()
    da.debug = debug
    da.initialize()
    da.setAnnealParameterMM(param)
    #print('\n params set',time.time()-st)
    result = da.doAnnealMM(args)
    #print(da._rpc_time())
    return result


#A predictor using DA.It can update the probability given constraints and probability.
class Predictor_da:
    def __init__(self,data,portion,a=60,b=30,c=50,N=4096,seed=None):
        #data: probability of categories
        self.a = a
        self.b = b
        self.c = c
        self.user_number = data.shape[0]
        self.cat_num = data.shape[1]
        self.data = data
        self.N = N
        self.portion = portion
        self.da_satisfied = 0
        assert(self.user_number*self.cat_num<=N)
        self.result = np.empty((self.user_number,),dtype=int)
        self.DA_hit_result = np.full((self.user_number,),-1)
#Start predict the correct answer
    def run_predict(self):
        prob_matrix = self.data*100
        prob_vector = np.array(prob_matrix.ravel().tolist()+[0]*(self.N-self.user_number*self.cat_num),dtype=int)
        hori_mat,vert_mat,hori_vector,vert_vector,hori_const,vert_const = generate_da_constraints(self.portion,self.N,self.user_number,self.cat_num)
        self.res = self.run_da(hori_mat,vert_mat,hori_vector,vert_vector,prob_vector,vert_const+hori_const)
        self.new_prob = self.update_da_prob(self.res)
            
    def run_da(self,hori_matrix,vert_matrix,hori_vector,vert_vector,prob_vector,weight_const):
        args,params = setup(num_bit=self.N,weight=-(self.a*hori_matrix+self.b*vert_matrix),bias=(-(self.a*hori_vector+self.b*vert_vector-self.c*prob_vector)),
                           constant=int(0))
                           #constant=int(weight_const))
        #print(args['lf_i_n'])
        self.args,self.params = args,params
        res = run(args,params)
        return res
#Update the probability after receiving the DA result
    def update_da_prob(self,res):
        state_min = [(ord(c)-ord('0')) for c in res['state_min_o_n'][0]]
        rrr = np.array(state_min[:self.user_number*self.cat_num]).reshape(self.user_number,self.cat_num)
        self.da_qubits_result = rrr
        cat_num = [0]*self.cat_num
        new_probs = np.array(self.data)
        for row_idx in range(self.user_number):            
            s = sum(rrr[row_idx])
            
            if s==1: #exact one answer
                max_cat = np.argmax(rrr[row_idx])
                new_probs[row_idx][max_cat]+=1
                self.DA_hit_result[row_idx] = max_cat
                self.da_satisfied+=1
            elif s>1: #multiple answers
                new_probs[row_idx][np.where(rrr[row_idx]==1,True,False)]+=1
        return new_probs

#User greedy postprocess.
def greedy_user(probs,portion):
    cat_num = probs.shape[1]

    assert(len(portion)==cat_num)
    cat_num = np.zeros((cat_num,))
    prediction = []
    for row in probs:
        #sort the probs
        idx_arr = np.argsort(row)[::-1]
        #try each cat (from the largest probability)
        for i in idx_arr:
            if cat_num[i]<portion[i]:
                cat_num[i]+=1
                prediction.append(i)
                break
    return prediction

#Cate greedy postprocess. No use in real scenario. It is just kept for comparison.
def greedy_cat(probs,portion):
    cat_num = probs.shape[1]
    user_num = probs.shape[0]
    #initialize all prediction with -1
    prediction = np.full((user_num,),-1)
    portion = portion.astype(int)
    #print(sum(portion))
    for i in range(len(portion)):
        #for each portion , get the order of user [sort decending]
        sorted_tmp = np.argsort(probs[:,i])[::-1]
        k = 0
        for j in range(portion[i]):
            #user_row_number is most probable
            user_row_number = sorted_tmp[k]
            #find the next un_predicted user
            while(prediction[user_row_number]!=-1):
                k+=1
                user_row_number = sorted_tmp[k]
            prediction[user_row_number]=i
    return prediction

#Because portion is decimal, we need to transfer it into integers because the number of users is integer.
def normalize_portion(portions,users=1):
    portions = np.array(portions,dtype=float)
    portions/=sum(portions)
    portions*=users
    portion_int = portions.astype(int)
    reminds = portions-portion_int
    x = users-sum(portion_int)
    s = np.argsort(reminds)[::-1]
    i=0
    
    while (x>0):
        portion_int[s[i]]+=1
        i+=1
        x-=1
    return portion_int

def prec(result,gt):
    total = 0
    hit = 0
    for i in range(len(result)):
        
        if gt[i]==-2:
            continue
        else:

            total+=1
            if result[i]==gt[i]:
                hit+=1
    if total==0:
        return 0
    return hit/total
import datetime

#   This is the iterator that generate the iterator of time.
#   It will function correctly if the 24 hours is divisible by the timestr (e.g. 4h, 1h , 20 min)
#   Please note that if the 24 hour cannot be divisible by the timestr (e.g. 50min), the final time slot
# may not be generated. [This is a bug]. In this case, please edit the calculation of self.limit (add 1 to it.)
class Time_Iterator:
    def __init__(self,minute=20):
        self.minute=minute
        st=datetime.time(0,0)
        self.cnt = st
    def __iter__(self):
        
        self.limit = 60*24//self.minute
        #print(self.limit)
        self.i = 0
        return self
    def __next__(self):
        if self.i<self.limit:
            result = self.cnt.strftime('%H%M')
            self.cnt=(datetime.datetime.combine(datetime.date(1,1,1),self.cnt)+datetime.timedelta(minutes=self.minute)).time()
            self.i+=1
            return result
        elif self.i==self.limit and self.cnt.hour!=0 and self.cnt.minute!=0:
            self.i+=1
            return self.cnt.strftime('%H%M')
        else:
            raise StopIteration

import sys
#minutestr = 20

minutestr = int(sys.argv[3])

folder = sys.argv[1]


constraint_file = open(r'{}/category_distribution.csv'.format(folder))
constraints,categories = read_constraints(constraint_file)

logname = sys.argv[2]
result_file = open('log{}_withgreedy.txt'.format(logname),'w')
import time
iterator = Time_Iterator(int(minutestr))
print('time\tamount\tdahit\tdaprec\tusergreedy\tdausergreedy\tcategreedy\tdacat\tcnttime',file=result_file)

for i,timestr in enumerate(iterator):
    f = open(r'{}/cat_prob_1107{}.csv'.format(folder,timestr))
    file_data = read_file(f,categories[i])
    np.random.seed(10)
    np.random.shuffle(file_data)
    remaining = file_data
    num_cat = file_data.shape[1]-1
    user_amount = file_data.shape[0]
#             if user_amount*num_cat<1024:
#                 NUM_BITS = 1024
#             elif user_amount*num_cat<2048:
#                 NUM_BITS = 2048
#             elif user_amount*num_cat<4096:
#                 NUM_BITS = 4096
#             else:
#                 NUM_BITS = 8192
    NUM_BITS=2048
#             print(NUM_BITS,'bits')
    batch_size = NUM_BITS//num_cat
    result_cat = np.empty((0,))
    result_user = np.empty((0,))
    result_da_cat = np.empty((0,))
    result_da_user = np.empty((0,))
    DA_hit_result = np.empty((0,))
    da_satisfied = 0
    gt = file_data[:,num_cat]
    st_t = time.time()
    for j in range(file_data.shape[0]//batch_size+1):
        st = time.time()
        probs = file_data[j*batch_size:(j+1)*batch_size,:num_cat]
        normalized_portion = normalize_portion(constraints[i],users=probs.shape[0])
        result_cat = np.concatenate((result_cat,greedy_cat(portion=normalized_portion,probs=probs)),axis=0)
        result_user = np.concatenate((result_user,greedy_user(portion=normalized_portion,probs=probs)),axis=0)
        #print(result_cat)
        pda = Predictor_da(probs,normalized_portion,a=5,b=10,c=1,N=NUM_BITS)
        #print(NUM_BITS,timestr,j,probs.shape[0],sep='\t',end='\t')
        pda.run_predict()
        new_probs = pda.new_prob
        DA_hit_result = np.concatenate((DA_hit_result,pda.DA_hit_result))
        da_satisfied+=pda.da_satisfied
        result_da_cat = np.concatenate((result_da_cat,greedy_cat(portion=normalized_portion,probs=new_probs)),axis=0)
        result_da_user = np.concatenate((result_da_user,greedy_user(portion=normalized_portion,probs=new_probs)),axis=0)
        #print(time.time()-st)
        #print('round {}, using {} sec'.format(j,time.time()-st))
    prec_cat = prec(result_cat,gt) #sum(np.equal(result_cat,gt))/len(gt)
    prec_user = prec(result_user,gt) #sum(np.equal(result_user,gt))/len(gt)
    prec_dacat = prec(result_da_cat,gt) #sum(np.equal(result_da_cat,gt))/len(gt)
    prec_dauser = prec(result_da_user,gt) #sum(np.equal(result_da_user,gt))/len(gt)
    prec_dahit = prec(DA_hit_result,gt)#sum(np.equal(DA_hit_result,gt))/len(gt)
    ed_t = time.time()
    print("{}\t{}\t{:.2f}\t{:.3f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.1f}".format(timestr,len(gt),da_satisfied/len(gt),prec_dahit,prec_user,prec_dauser,prec_cat,prec_dacat,ed_t-st_t),file=result_file)
    result_file.flush()



