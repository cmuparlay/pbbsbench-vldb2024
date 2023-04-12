#!/bin/bash
# EXPORT LIST
# dataset
# scale
# dtype
# dist
# file_in
# file_q
# file_gt
# lsh
# rr
# th
# L
# rot
# K
# lastk
# lc
# (rad)
algo="FALCONN"
RESULT_DIR=${RESULT_PREFIX}/${algo}/$dataset/L${L}_rot${rot}_K${K}_${lsh}${lastk}_${dist}

#set -x
date

mkdir -p $RESULT_DIR

echo "Running for the first ${scale} million points on ${dataset}"
param_basic="-n $((scale*1000000)) -type ${dtype}"
param_building="-dist ${dist} -in ${file_in} -lsh ${lsh} -K ${K} -lastk ${lastk}"
param_query="-q ${file_q} -g ${file_gt} -r ${rr} -th ${th} -l ${L} -lc ${lc}"
param_other="-rot ${rot}"
if [ -n "$rad" ]; then
	param_other="${param_other} -rad ${rad}"
fi

LOG_PATH=${RESULT_DIR}/${scale}M.log
echo "./calc_recall ${param_basic} ${param_building} ${param_query} ${param_other} > ${LOG_PATH} 2>&1"
`which time` -v ./calc_recall ${param_basic} ${param_building} ${param_query} ${param_other} > ${LOG_PATH} 2>&1

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