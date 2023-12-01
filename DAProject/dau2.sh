#This file is for activate the dau2 environment, including library and paths
export PYTHONPATH=/opt/FJSVda/lib/python3.7
export LD_LIBRARY_PATH=/opt/FJSVda/lib/fjda_grpc
export FJDA_CA_FILE=/opt/FJSVda/lib/fjda_grpc/DA-CA.pem
export no_proxy="localhost,127.0.0.0/8,::1,da.labs.fujitsu.com,default.svc.cluster.local,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12"
#export FJDA_SERVER=dau2-04-0.da.labs.fujitsu.com
export FJDA_SERVER=dau2-05-0.da.labs.fujitsu.com
conda activate my-python37
