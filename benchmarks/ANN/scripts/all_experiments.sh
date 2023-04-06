#!/bin/bash
cd ~/pbbsbench/benchmarks/ANN/scripts
#nearest neighbor search experiments
bash vamana.sh
bash hcnng.sh
bash pynndescent.sh
bash hnsw.sh
bash FALCONN.sh
#range search experiments
cd ~/pbbsbench/benchmarks/rangeSearch/scripts
bash all_experiments.sh
#all experiments with FAISS
cd ~/pbbsbench/benchmarks/ANN/scripts
bash run.sh



