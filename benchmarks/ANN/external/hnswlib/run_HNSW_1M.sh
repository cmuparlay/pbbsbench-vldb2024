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
export ef=5,7,10,15,20,30,50,75,100,125,250,500
export beta=1
export th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.85,0.95,0.99,0.999

export save_graph=0
export warmup=1
export thread=`nproc`
export limit_eval=1

threadset=(96 1 2 8 24 48 16 4)

dataset=BIGANN
dtype=uint8
dist=L2

dataset_base=/ssd1/data/bigann
file_in=${dataset_base}/base.1B.u8bin.crop_nb_1000000:u8bin
file_q=${dataset_base}/query.public.10K.u8bin:u8bin

m=32
efc=128
alpha=0.82

scale=1
file_gt=${dataset_base}/bigann-1M:ubin
for thread in ${threadset[@]}; do
	bash run_HNSW_single.sh
done

exit
m=50
efc=200

scale=1
file_gt=${dataset_base}/bigann-1M:ubin
for thread in ${threadset[@]}; do
	bash run_HNSW_single.sh
done

