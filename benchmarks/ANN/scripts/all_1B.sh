#!/bin/bash

P=/ssd1/data
G=/ssd1/results
BP=$P/bigann
BG=$G/bigann
SP=$P/MSSPACEV1B
SG=$G/MSSPACEV1B
TP=$P/text2image1B
TG=$G/text2image1B
FP=$P/FB_ssnpp
FG=$G/FB_ssnpp

mkdir -p $BG
mkdir -p $SG
mkdir -p $TG
mkdir -p $FG

#Vamana
cd ~/pbbsbench/benchmarks/ANN/vamana
make clean all
./neighbors -R 64 -L 128 -o $BG/1B_vamana_64_128 -q $BP/query.public.10K.u8bin -c $BP/GT.public.1B.ibin -res $G/bigann_vamana.csv -f bin -t uint8 $BP/base.1B.u8bin
./neighbors -R 64 -L 128 -o $SG/1B_vamana_64_128 -q $SP/query.i8bin -c $SP/public_query_gt100.bin -res $G/spacev_vamana.csv -f bin -t int8 $SP/spacev1b_base.i8bin
./neighbors -a .9 -R 64 -L 128 -o $TG/1B_vamana_64_128 -q $TP/query.public.100K.fbin -c $TP/t2i_new_groundtruth.public.100K.bin -res $G/t2i_vamana.csv -f bin -t float -D 1 $TP/base.1B.fbin

#HCNNG
cd ~/pbbsbench/benchmarks/ANN/HCNNG
make clean all
./neighbors -a 1000 -R 3 -L 30 -b 1 -o $BG/1B_HCNNG_30 -q $BP/query.public.10K.u8bin -c $BP/GT.public.1B.ibin -res $O/bigann_HCNNG.csv -f bin -t uint8 $BP/base.1B.u8bin
./neighbors -a 1000 -R 3 -L 50 -b 1 -o $SG/1B_HCNNG_50 -q $SP/query.i8bin -c $SP/public_query_gt100.bin -res $O/spacev_HCNNG.csv -f bin -t int8 $SP/spacev1b_base.i8bin
./neighbors -a 1000 -R 3 -L 30 -b 1 -o $TG/1B_HCNNG_30 -q $TP/query.public.100K.fbin -c $TP/t2i_new_groundtruth.public.100K.bin -res $TG/t2i_HCNNG.csv -f bin -t float -D 1 $TP/base.1B.fbin

#range search, three algos
cd ~/pbbsbench/benchmarks/rangeSearch/vamana
make clean all
./range -a 1.2 -R 150 -L 400 -q $FP/FB_ssnpp_public_queries.u8bin -o $FG/1B_150_400 -c $FP/FB_ssnpp_public_queries_1B_GT.rangeres -res $G/ssnpp_vamana.csv $FP/FB_ssnpp_database.u8bin
cd ~/pbbsbench/benchmarks/rangeSearch/HCNNG
make clean all
./range -a 1000 -R 3 -L 50 -b 1 -q $FP/FB_ssnpp_public_queries.u8bin -o $FG/1B_3_50 -c $FP/FB_ssnpp_public_queries_1B_GT.rangeres -res $G/ssnpp_hcnng.csv $FP/FB_ssnpp_database.u8bin

cd ~/pbbsbench/benchmarks/ANN/scripts

#FALCONN
bash FALCONN/FALCONN_1B.sh

#HNSW
bash hnsw/hnsw_1B.sh

#FAISS
bash FAISS/FAISS_1B.sh