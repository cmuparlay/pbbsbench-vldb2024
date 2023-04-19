export RESULT_PREFIX="/ssd1/results"

export dataset=
export dtype=
export dist=
export m=
export efc=
export alpha=

export scale=
export file_in=
export file_q=
export file_gt=

export rr=10
export ef=15,20,30,50,75,100,125,250,500
export beta=1
export th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999

export warmup=0
export save_graph=1
export thread=`nproc`
export limit_eval=1

cd ~/pbbsbench/benchmarks/ANN/HNSW
make calc_recall

P=/ssd1/data
G=/ssd1/results

#-------------------------------------------------
# BIGANN
BP=$P/bigann
BG=$G/bigann
dataset=BIGANN
dtype=uint8
dist=L2
m=32
efc=128
alpha=0.82
file_in=$BP/base.1B.u8bin:u8bin
file_q=$BP/query.public.10K.u8bin:u8bin

scale=1000
file_gt=$BP/bigann-1B:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999
bash run_HNSW_single.sh

#-------------------------------------------------
#MSSPACEV
SP=$P/MSSPACEV1B
SG=$G/MSSPACEV1B
dataset=MSSPACEV
dtype=int8
dist=L2
m=32
efc=128
alpha=0.83
file_in=$SP/spacev1b_base.i8bin:i8bin
file_q=$SP/query.i8bin:i8bin

scale=1000
file_gt=$SP/msspacev-1B:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95
bash run_HNSW_single.sh

#-------------------------------------------------
#TEXT2IMAGE
TP=$P/text2image1B
TG=$G/text2image1B
dataset=YandexT2I
dtype=float
dist=ndot
m=32
efc=128
alpha=1.1
file_in=$TP/base.1B.fbin:fbin
file_q=$TP/query.public.100K.fbin:fbin

scale=1000
file_gt=$TP/text2image-1B:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
bash run_HNSW_single.sh

#-------------------------------------------------
#FB_SSNPP
P=$FP/FB_ssnpp
R=$FG/FB_ssnpp
dataset=FB_ssnpp
dtype=uint8
dist=L2
m=75
efc=400
alpha=0.82
file_in=$P/FB_ssnpp_database.u8bin:u8bin
file_q=$P/FB_ssnpp_public_queries.u8bin:u8bin
export rad=96237

scale=1000
file_gt=$P/FB_ssnpp_public_queries_1B_GT.rangeres:irange
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7
bash run_HNSW_single.sh