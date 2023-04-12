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
./neighbors -R 64 -L 128 -o $BG/100M_vamana_64_128 -q $BP/query.public.10K.u8bin -c $BP/bigann-100M -res $G/bigann_vamana.csv -f bin -t uint8 $BP/base.1B.u8bin.crop_nb_100000000
./neighbors -R 64 -L 128 -o $SG/100M_vamana_64_128 -q $SP/query.i8bin -c $SP/msspacev-100M -res $G/spacev_vamana.csv -f bin -t int8 $SP/spacev1b_base.i8bin.crop_nb_100000000
./neighbors -a .9 -R 64 -L 128 -o $TG/100M_vamana_64_128 -q $TP/query.public.100K.fbin -c $TP/text2image-100M -res $G/t2i_vamana.csv -f bin -t float -D 1 $TP/base.1B.fbin.crop_nb_100000000

#HCNNG
cd ~/pbbsbench/benchmarks/ANN/HCNNG
make clean all
./neighbors -a 1000 -R 3 -L 30 -b 1 -o $BG/100M_HCNNG_30 -q $BP/query.public.10K.u8bin -c $BP/bigann-100M -res $O/bigann_HCNNG.csv -f bin -t uint8 $BP/base.1B.u8bin.crop_nb_100000000
./neighbors -a 1000 -R 3 -L 50 -b 1 -o $SG/100M_HCNNG_50 -q $SP/query.i8bin -c $SP/msspacev-100M -res $O/spacev_HCNNG.csv -f bin -t int8 $SP/spacev1b_base.i8bin.crop_nb_100000000
./neighbors -a 1000 -R 3 -L 30 -b 1 -o $TG/100M_HCNNG_30 -q $TP/query.public.100K.fbin -c $TP/text2image-100M -res $TG/t2i_HCNNG.csv -f bin -t float -D 1 $TP/base.1B.fbin.crop_nb_100000000

#PyNNDescent
cd ~/pbbsbench/benchmarks/ANN/pyNNDescent
make clean all
./neighbors -R 40 -L 100 -a 10 -d 1.2 -o $BG/100M_pynn_40 -q $BP/query.public.10K.u8bin -c $BP/bigann-100M -res $G/bigann_pynn.csv -f bin -t uint8 $BP/base.1B.u8bin.crop_nb_100000000
./neighbors -R 60 -L 100 -a 10 -d 1.2 -o $SG/100M_pynn_60 -q $SP/query.i8bin -c $SP/msspacev-100M -res $G/spacev_pynn.csv -f bin -t int8 $SP/spacev1b_base.i8bin.crop_nb_100000000
./neighbors -R 60 -L 100 -a 10 -d .9 -o $TG/100M_pynn_60 -q $TP/query.public.100K.fbin -c $TP/text2image-100M -res $G/t2i_pynn.csv -f bin -t float -D 1 $TP/base.1B.fbin.crop_nb_100000000

#range search, three algos
cd ~/pbbsbench/benchmarks/rangeSearch/vamana
make clean all
./range -a 1.2 -R 150 -L 400 -q $FP/FB_ssnpp_public_queries.u8bin -o $FG/100M_150_400 -c $FP/ssnpp-100M -res $G/ssnpp_vamana.csv $FP/FB_ssnpp_database.u8bin.crop_nb_100000000
cd ~/pbbsbench/benchmarks/rangeSearch/HCNNG
make clean all
./range -a 1000 -R 3 -L 50 -b 1 -q $FP/FB_ssnpp_public_queries.u8bin -o $FG/100M_3_50 -c $FP/ssnpp-100M -res $G/ssnpp_hcnng.csv $FP/FB_ssnpp_database.u8bin.crop_nb_100000000
cd ~/pbbsbench/benchmarks/rangeSearch/pyNNDescent
make clean all
./range -R 60 -L 1000 -a 20 -d 1.4 -q $FP/FB_ssnpp_public_queries.u8bin -o $FG/100M_60 -c $FP/ssnpp-100M -res $G/ssnpp_pynn.csv $FP/FB_ssnpp_database.u8bin.crop_nb_100000000

cd ~/pbbsbench/benchmarks/ANN/scripts

#FALCONN
bash FALCONN/FALCONN_100M.sh

#HNSW
bash hnsw/hnsw_100M.sh

#FAISS
bash FAISS/FAISS_100M.sh