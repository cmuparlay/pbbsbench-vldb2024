CXX = g++
CXXFLAGS_BASIC = -std=c++17 -Wall -Wextra -pthread -I../bench
CXXFLAGS_DEBUG = -O0 -g
CXXFLAGS_RELEASE = -O3 -march=native
CXXFLAGS_HDF5 = -lhdf5 -lhdf5_cpp

gen_model: gen_model.cpp HNSW.hpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) $(CXXFLAGS_HDF5) -o $@

GloVe: test_glove.cpp HNSW.hpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) $(CXXFLAGS_HDF5) -o $@

calc_dist: calc_dist.cpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) $(CXXFLAGS_HDF5) -o $@

calc_recall: calc_recall.cpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) $(CXXFLAGS_HDF5) -Wl,--unresolved-symbols=ignore-all -fdiagnostics-color=always -o $@

eval_graph: eval_graph.cpp HNSW.hpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) $(CXXFLAGS_HDF5) -o $@
