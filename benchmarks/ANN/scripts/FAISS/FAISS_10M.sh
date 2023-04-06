#!/bin/bash
cd ~/big-ann-benchmarks
rm results
mkdir /ssd1/results/FAISSresults10M
ln -s /ssd1/results/FAISSresults10M results
P=~/pbbsbench/benchmarks/ANN/scripts/FAISS
R=/ssd1/results/FAISSresults10M

nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "bigann-10M" > $R/bigann-10M.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "msspacev-10M" > $R/msspacev-10M.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "text2image-10M" > $R/text2image-10M.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "ssnpp-10M" > $R/ssnpp-10M.log

sudo chmod -R 777 results/
python data_export.py --output /ssd1/results/FAISS_res_10M.csv