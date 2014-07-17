CC = g++
# -march=native will compile for correct simd
CFLAGS = -march=native -O3 -Wall

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g -Wall
endif

all: test swimd_aligner

swimd_aligner: Swimd.o ScoreMatrix.o swimd_aligner.cpp
	$(CC) $(CFLAGS) Swimd.o ScoreMatrix.o swimd_aligner.cpp -o swimd_aligner

ScoreMatrix.o: ScoreMatrix.cpp ScoreMatrix.hpp
	$(CC) $(CFLAGS) -c ScoreMatrix.cpp

test: Swimd.o test.cpp
	$(CC) $(CFLAGS) Swimd.o test.cpp -o test

Swimd.o: Swimd.cpp Swimd.h
	$(CC) $(CFLAGS) -c Swimd.cpp

clean:
	rm *.o test swimd_aligner
