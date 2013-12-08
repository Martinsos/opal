CC = g++
# -march=native will compile for correct simd
CFLAGS = -march=native

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g
endif
ifeq ($(DEBUG), 0)
	CFLAGS += -O3
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
