CXX = g++
#CXX = clang++
CXXFLAGS_BASIC = -std=c++17 -Wall -Wextra -pthread -march=native -I../bench 
ifeq (DEBUG, $(MODE))
	CXXFLAGS_BASIC += -O0 -g
else
	CXXFLAGS_BASIC += -O3
endif

CXXFLAGS_HDF5 =
ifeq (ON, $(HDF5))
	CXXFLAGS_HDF5 = -DSUPPORT_HDF5 -lhdf5 -lhdf5_cpp
endif

CXXFLAGS_JEMALLOC =
ifneq (, $(shell which jemalloc-config))
	LIBDIR_JEMALLOC := $(shell jemalloc-config --libdir)
	CXXFLAGS_JEMALLOC := -L$(LIBDIR_JEMALLOC) -ljemalloc -Wl,-rpath=$(LIBDIR_JEMALLOC)
endif

CXXFLAGS_IGNORE_SYMBOL = -Wl,--unresolved-symbols=ignore-all

.PHONY: all

all: calc_recall gen_model eval_graph

gen_model: gen_model.cpp HNSW.hpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_HDF5) $(CXXFLAGS_JEMALLOC) $(CXXFLAGS_IGNORE_SYMBOL) -o $@

calc_recall: calc_recall.cpp HNSW.hpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_HDF5) $(CXXFLAGS_JEMALLOC) $(CXXFLAGS_IGNORE_SYMBOL) -o $@

eval_graph: eval_graph.cpp HNSW.hpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_HDF5) $(CXXFLAGS_JEMALLOC) -o $@
