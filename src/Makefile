CXX = g++
# -march=native will compile for correct simd
CXXFLAGS = -std=c++11 -march=native -Wall -faligned-new

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -fno-omit-frame-pointer
else
	CXXFLAGS += -O3
endif

all: test opal_aligner

opal_aligner: opal.o ScoreMatrix.o opal_aligner.cpp
	$(CXX) $(CXXFLAGS) opal.o ScoreMatrix.o opal_aligner.cpp -o opal_aligner

ScoreMatrix.o: ScoreMatrix.cpp ScoreMatrix.hpp
	$(CXX) $(CXXFLAGS) -c ScoreMatrix.cpp

test: opal.o ScoreMatrix.o test.cpp
	$(CXX) $(CXXFLAGS) opal.o ScoreMatrix.o test.cpp -o test

opal.o: opal.cpp opal.h
	$(CXX) $(CXXFLAGS) -c opal.cpp

clean:
	rm *.o test opal_aligner
