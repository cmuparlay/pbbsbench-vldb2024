export RESULT_PREFIX="temp/OOD"

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
export ef=10,15,20,30,50,75,100,250,500,1000,2000
export beta=1
export th=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999

export warmup=0
export save_graph=0
export thread=`nproc`
export limit_eval=0

#-------------------------------------------------
dataset=YandexT2I
dtype=float
dist=ndot

dir_dataset=/data8/zshen055/Yandex-Text-to-Image
#dir_dataset=/data2/zshen055/ANN/ANN_SIFT1B
#file_in=$dir_dataset/base.1B.u8bin:u8bin
#file_q=$dir_dataset/query.public.10K.u8bin:u8bin
file_in=$dir_dataset/base.1B.fbin:fbin
file_q=$dir_dataset/query.public.100K.fbin:fbin

m=32
efc=128
alpha=1.1

scale=1
file_gt=$dir_dataset/text2image-1M:ubin
#bash run_HNSW_single.sh

scale=10
file_gt=$dir_dataset/text2image-10M:ubin
bash run_HNSW_single.sh

scale=100
file_gt=$dir_dataset/text2image-100M:ubin
#bash run_HNSW_single.sh

scale=1000
file_gt=$dir_dataset/text2image-1B:ubin
#bash run_HNSW_single.sh
