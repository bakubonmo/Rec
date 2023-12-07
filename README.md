# DAProject
For commercial companies, tuning advertisement delivery to achieve a high conversion rate (CVR) is crucial 
for improving advertising effectiveness. Because advertisers use demand-side platforms (DSP) to deliver a 
certain number of ads within a fixed period, it is challenging for DSP to maximize CVR while satisfying 
delivery constraints such as the number of delivered ads in each category. Although previous research aimed 
to optimize the combinational problem under various constraints, its periodic updates remained an open 
question because of its time complexity. Our work is the first attempt to adopt digital annealers (DAs), which 
are quantum-inspired computers manufactured by Fujitsu Ltd., to achieve real-time periodic ad optimization. 
With periodic optimization in a short time, we have much chance to increase ad recommendation precision. 
First, we exploit each user’s behavior according to his visited web pages and then predict his CVR for each 
ad category. Second, we transform the optimization problem into a quadratic unconstrained binary 
optimization model applying to the DA. The experimental evaluations on real log data show that our proposed 
method improves accuracy score from 0.237 to 0.322 while shortening the periodic advertisement 
recommendation from 526s to 108s (4.9 times speed-up) in comparison with traditional algorithms.

## Figure
![image](https://github.com/bakubonmo/Rec/assets/122580605/d2d2f1b9-9fd9-49a4-a05f-3b8c25a1c5d1)




## Prerequisites
numpy<br>
Fujistu Daigital annealer(DA)


## How to Start
To finish an entire offline experiment: <br>
Firstly, activate virtual environment (by ```source dau2.sh``` or other script) and run newdata_with_greedy_1124.py. <br>
Secondly, open the output log file and copy all the results. <br>
Lastly, paste the results into excel file on your PC(you can choose sheet of corresponding time slot or create new sheet), the formulas will automatically give the result.



<br><br><br>
# GN-GCN
Point-of-interest (POI) recommendation helps users filter information and discover their interests. In recent years, 
graph convolution network (GCN)– based methods have become state-of-the-art algorithms for improving recommendation performance. 
Especially integrating GCN with multiple information, such as geographical information, is a promising way to achieve better performance;
however, it tends to increase the number of trainable parameters, resulting in the difficulty of model training to reduce the performance. 
In this study, we mine users’ active areas and extend the definition of neighbors in GCN, called active area neighbors. 
Our study is the first attempt to integrate geographic information into a GCN POI recommendation system without increasing 
the number of trainable parameters and maintaining the ease of training. The experimental evaluation confirms that compared 
with the state-of-the-art lightweight GCN models, our method improves Recall@10 from 0.0562 to 0.0590 (4.98%) on Yelp dataset and from 0.0865 to 0.0898 (3.82%) on Gowalla dataset.


## Figure
![image](https://github.com/bakubonmo/Rec/assets/122580605/3d673f5a-4058-458c-9655-32135adf8b30)




## Prerequisites
Python 3.6 <br>
Pytorch 1.11.0



## How to Start
Please run geo2npy to preprocess <br>
Please run the file "train" to train the model  <br>
Please run the file "test" to test the model <br>




<br><br><br>
# EPT-GCN
In location-based social networks (LBSNs), point-of-interest (POI) recommendation systems help users
identify unvisited POIs by filtering large amounts of information. Accurate POI recommendations can
effectively improve user satisfaction and save time in finding POIs. In recent years, the graph convolution
network (GCN) technique, which enhances the representational ability of neural networks by learning the
embeddings of users and items, has been widely adopted in recommendation systems to improve accuracy. 
Combining GCN with various information, such as time and geographical information, can further
improve recommendation performance. However, existing GCN-based techniques simply adopt time
information by modeling users’ check-in sequences, which is insufficient and ignores users’ time-based
high-order connectivity. Note that time-based high-order connectivity refers to the relationship between
indirect neighbors with similar preferences in the same time slot. In this paper, we propose a new 
time-aware GCN model to extract rich collaborative signals contained in time information. Our work is the first
to divide user check-ins into multiple subgraphs, i.e., time slots, based on time information. We further
propose an edge propagation module to adjust edge affiliation, where edges represent check-ins, 
to propagate user’s time-based preference to multiple time slots. The propagation module is based on 
an unsupervised learning algorithm and does not require additional ground-truth labels. Experimental results
confirm that our method outperforms state-of-the-art GCN models in all baselines, improving Recall@5
from 0.0803 to 0.0874 (8.84%) on the Gowalla dataset and from 0.0360 to 0.0388 (7.78%) on the New
York dataset. The proposed subgraph mining technique and novel edge-based propagation module have
high scalability and can be applied to other subgraph construction models.


## Figure
![image](https://github.com/bakubonmo/Rec/assets/122580605/8f16ed5f-ac2d-467b-9e1a-6c074e5f119c)



## Prerequisites
Python 3.6 <br>
Pytorch 1.11.0


## How to Start
Please run the file "train" to train the model <br>
Please run the file "test" to test the model <br>


