CXX = g++
# -march=native will compile for correct simd
CXXFLAGS = -std=c++11 -march=native -Wall

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g
else
	CXXFLAGS += -O3
endif

all: test swimd_aligner

swimd_aligner: Swimd.o ScoreMatrix.o swimd_aligner.cpp
	$(CXX) $(CXXFLAGS) Swimd.o ScoreMatrix.o swimd_aligner.cpp -o swimd_aligner

ScoreMatrix.o: ScoreMatrix.cpp ScoreMatrix.hpp
	$(CXX) $(CXXFLAGS) -c ScoreMatrix.cpp

test: Swimd.o ScoreMatrix.o test.cpp
	$(CXX) $(CXXFLAGS) Swimd.o ScoreMatrix.o test.cpp -o test

Swimd.o: Swimd.cpp Swimd.h
	$(CXX) $(CXXFLAGS) -c Swimd.cpp

clean:
	rm *.o test swimd_aligner
