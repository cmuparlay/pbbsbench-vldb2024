export RESULT_PREFIX="."

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
export ef=10,15,20,30,50,75,100
export beta=1
export th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9

export save_graph=1
export warmup=1
export thread=`nproc`
export limit_eval=1
export load_graph=1

make calc_recall

P=/ssd1/data
G=/ssd1/results

#-------------------------------------------------
BP=$P/bigann
BG=$G/bigann
# BIGANN: two settings
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
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
bash run_HNSW_single.sh


#-------------------------------------------------
SP=$P/MSSPACEV1B
SG=$G/MSSPACEV1B
#MSSPACEV: two settings
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
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
bash run_HNSW_single.sh

