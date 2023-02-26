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

export rr=50
export ef=500
export beta=1
export th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99

export rad=96237

export warmup=0
export save_graph=1

make calc_recall

P=/ssd1/data/FB_ssnpp
R=/ssd1/results/FB_ssnpp

#-------------------------------------------------
dataset=FB_ssnpp
dtype=uint8
dist=L2

alpha=0.82
file_in=$P/FB_ssnpp_database.u8bin:u8bin
file_q=$P/FB_ssnpp_public_queries.u8bin:u8bin


scale=1
file_gt=$P/ssnpp-1M:irange
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99

m=64
efc=256
bash run_HNSW_single.sh

m=75
efc=400
bash run_HNSW_single.sh


scale=10
file_gt=$P/ssnpp-10M:irange
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95

m=64
efc=256
bash run_HNSW_single.sh

m=75
efc=400
bash run_HNSW_single.sh


scale=100
file_gt=$P/ssnpp-100M:irange
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8

m=64
efc=256
bash run_HNSW_single.sh

m=75
efc=400
bash run_HNSW_single.sh


scale=1000
file_gt=$P/FB_ssnpp_public_queries_1B_GT.rangeres:irange
th=0.1,0.2,0.3,0.4,0.5,0.6,0.7

m=75
efc=400
bash run_HNSW_single.sh

