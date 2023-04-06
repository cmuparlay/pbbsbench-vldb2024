#!/bin/bash
cd ~/big-ann-benchmarks
rm results
mkdir /ssd1/results/FAISSresults1B
ln -s /ssd1/results/FAISSresults1B results
P=~/pbbsbench/benchmarks/ANN/scripts/FAISS
R=/ssd1/results/FAISSresults1B

nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "bigann-1B" > $R/bigann-1B.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "msspacev-1B"  > $R/msspacev-1B.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "text2image-1B" > $R/text2image-1B.log
nohup python3 run.py --definitions $P/ANN.yaml --algorithm faiss-t1 --dataset "ssnpp-1B" > $R/ssnpp-1B.log

sudo chmod -R 777 results/
python data_export.py --output /ssd1/results/FAISS_res_1B.csv