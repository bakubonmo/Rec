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

##Prerequisites
numpy
Fujistu Daigital annealer(DA)


##How to Start
To finish an entire offline experiment: 
Firstly, activate virtual environment (by ```source dau2.sh``` or other script) and run newdata_with_greedy_1124.py. 
Secondly, open the output log file and copy all the results. 
Lastly, paste the results into excel file on your PC(you can choose sheet of corresponding time slot or create new sheet), the formulas will automatically give the result.


##Results


