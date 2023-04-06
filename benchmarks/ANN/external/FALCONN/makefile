CXX = g++
#CXX = clang++
DIR_PARLAY = ./parlaylib/include
DIR_FALCONN = ../../include
DIR_EIGEN = ../../../external/eigen
DIR_SERIALIZER = ../../../external/simple-serializer
DIR_NLOHMANN = ../../../external/nlohmann
DIR_INCLUDE = $(DIR_PARLAY) $(DIR_FALCONN) $(DIR_EIGEN) $(DIR_SERIALIZER) $(DIR_NLOHMANN)

CXXFLAGS_BASIC = -std=c++17 -Wall -Wextra -march=native -pthread $(DIR_INCLUDE:%=-I%)
ifeq (DEBUG, $(MODE))
	CXXFLAGS_BASIC += -O0 -g
else
	CXXFLAGS_BASIC += -O3
endif
CXXFLAGS_JEMALLOC =
ifneq (, $(shell which jemalloc-config))
	LIBDIR_JEMALLOC := $(shell jemalloc-config --libdir)
	CXXFLAGS_JEMALLOC := -L$(LIBDIR_JEMALLOC) -ljemalloc -Wl,-rpath=$(LIBDIR_JEMALLOC)
endif

CXXFLAGS_HDF5 =
ifeq (ON, $(HDF5))
	CXXFLAGS_HDF5 = -DSUPPORT_HDF5 -lhdf5 -lhdf5_cpp
endif

calc_recall: calc_recall.cpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) $(CXXFLAGS_HDF5) $(CXXFLAGS_JEMALLOC) -o $@

default_param: default_param.cpp
	$(CXX) $< $(CXXFLAGS_BASIC) $(CXXFLAGS_RELEASE) -o $@