#!/bin/bash
cd ~/big-ann-benchmarks
rm results
mkdir /ssd1/results/FAISSresults100M
ln -s /ssd1/results/FAISSresults100M results
P=~/pbbsbench/benchmarks/ANN/scripts/FAISS
R=/ssd1/results/FAISSresults100M

nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "bigann-100M" > $R/bigann-100M.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "msspacev-100M"  > $R/msspacev-100M.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "text2image-100M" > $R/text2image-100M.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "ssnpp-100M" > $R/ssnpp-100M.log

sudo chmod -R 777 results/
python data_export.py --output /ssd1/results/FAISS_res_100M.csv