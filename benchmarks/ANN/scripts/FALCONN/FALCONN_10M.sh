export RESULT_PREFIX="/ssd1/results"

export dataset=
export dtype=float
export dist=L2

export scale=
export file_in=
export file_q=
export file_gt=

export lsh=cp
export rr=10
export th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999
export L=
export rot=1

export K=0
export lastk=0
export lc=

lc_10=5000,10000,20000,50000,100000,1000000

cd ~/pbbsbench/benchmarks/ANN/external/FALCONN
make calc_recall

P=/ssd1/data
G=/ssd1/results

#-------------------------------------------------
# BIGANN
BP=$P/bigann
BG=$G/bigann
dataset=BIGANN
file_in=$BP/base.1B.u8bin:u8bin
file_q=$BP/query.public.10K.u8bin:u8bin

L=30

scale=10
file_gt=$BP/bigann-10M:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99
lc=${lc_10}
bash run_FALCONN_single.sh


#-------------------------------------------------
#MSSPACEV
SP=$P/MSSPACEV1B
SG=$G/MSSPACEV1B
dataset=MSSPACEV
file_in=$SP/spacev1b_base.i8bin:i8bin
file_q=$SP/query.i8bin:i8bin

L=30

scale=10
file_gt=$SP/msspacev-10M:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95
lc=${lc_10}
bash run_FALCONN_single.sh

#-------------------------------------------------
#TEXT2IMAGE
TP=$P/text2image1B
TG=$G/text2image1B
dataset=YandexT2I
file_in=$TP/base.1B.fbin:fbin
file_q=$TP/query.public.100K.fbin:fbin
dist=ndot

L=30

scale=10
file_gt=$TP/text2image-10M:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
lc=${lc_10}
bash run_FALCONN_single.sh

export rad=96237
P=/ssd1/data/FB_ssnpp
R=/ssd1/results/FB_ssnpp
#-------------------------------------------------
dataset=FB_ssnpp
file_in=$P/FB_ssnpp_database.u8bin:u8bin
file_q=$P/FB_ssnpp_public_queries.u8bin:u8bin
dist=L2

L=30

scale=10
file_gt=$P/ssnpp-10M:irange
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95
lc=${lc_10}
bash run_FALCONN_single.sh