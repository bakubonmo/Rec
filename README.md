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
Firstly, Please download and put the dataset in the fold "data/datasetname/" <br>
Secondly, Please run geo2npy to process the dataset <br>
Thirdly, Please run the file "train" to train the model  <br>
Lastly, Please run the file "test" to test the model <br>




<br><br><br>


