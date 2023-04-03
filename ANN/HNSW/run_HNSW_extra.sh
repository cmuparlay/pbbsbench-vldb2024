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
export ef=15,20,30,50,75,100,125,250,500
export beta=1
export th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999

export save_graph=1

P=/ssd1/data
G=/ssd1/results

#-------------------------------------------------
BP=$P/bigann
BG=$G/bigann

dataset=BIGANN
dtype=uint8
dist=L2

alpha=0.82
file_in=$BP/base.1B.u8bin:u8bin
file_q=$BP/query.public.10K.u8bin:u8bin

m=50
efc=200

scale=1
file_gt=$BP/bigann-1M:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999
bash run_HNSW_single.sh

scale=10
file_gt=$BP/bigann-10M:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999
bash run_HNSW_single.sh

scale=100
file_gt=$BP/bigann-100M:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999
bash run_HNSW_single.sh

scale=1000
file_gt=$BP/bigann-1B:ubin
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999
bash run_HNSW_single.sh