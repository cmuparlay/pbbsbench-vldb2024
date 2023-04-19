#!/bin/bash
# EXPORT LIST
# dataset
# scale
# dtype
# dist
# m
# efc
# alpha
# file_in
# file_q
# file_gt
# ef
# rr
# beta
# th
# rad
# warmup
# save_graph
# thread
# limit_eval
algo="HNSW"
RESULT_DIR=${RESULT_PREFIX}/${algo}/$dataset/m${m}_efc${efc}_a${alpha}_${dist}_${dtype}_le${limit_eval}_thread${thread}

#set -x
date

mkdir -p $RESULT_DIR

echo "Running for the first ${scale} million points on ${dataset}"
param_basic="-n $((scale*1000000)) -type ${dtype} -dist ${dist}"
param_building="-ml 0.36 -m ${m} -efc ${efc} -alpha ${alpha} -b 2 -f 0 -in ${file_in}"
param_query="-q ${file_q} -g ${file_gt} -ef ${ef} -r ${rr} -beta ${beta} -th ${th}"
param_other="-w ${warmup} -le ${limit_eval}"
if [ -n "$rad" ]; then
	param_other="${param_other} -rad ${rad}"
fi
if [ $save_graph -ne 0 ]; then
	param_other="${param_other} -out ${RESULT_DIR}/${scale}M.bin"
fi

echo "Setting thread=${thread}"
export PARLAY_NUM_THREADS=${thread}

LOG_PATH=${RESULT_DIR}/${scale}M.log
echo "./calc_recall ${param_basic} ${param_building} ${param_query} ${param_other} > ${LOG_PATH} 2>&1"
./calc_recall ${param_basic} ${param_building} ${param_query} ${param_other} > ${LOG_PATH} 2>&1

shortname=$(echo $dataset | tr '[:upper:]' '[:lower:]')
if [[ "$dataset" == "BIGANN" ]]; then
	shortname="bigann"
elif [[ "$dataset" == "MSSPACEV" ]]; then
	shortname="spacev"
elif [[ "$dataset" == "YandexT2I" ]]; then
	shortname="t2i"
elif [[ "$dataset" == "FB_ssnpp" ]]; then
	shortname="ssnpp"
fi
CSV_PATH=${RESULT_PREFIX}/${shortname}_${algo}.csv
if [ -n "$rad" ]; then
	echo "python3 parse_range.py ${LOG_PATH} ${CSV_PATH}"
	python3 parse_range.py ${LOG_PATH} ${CSV_PATH}
else
	echo "python3 parse_kNN.py ${LOG_PATH} 0 ${CSV_PATH}"
	python3 parse_kNN.py ${LOG_PATH} 0 ${CSV_PATH}
fi