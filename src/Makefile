CC = g++
CFLAGS = -march=native -O3 # -march=native will compile for correct simd
OBJECTS = Swimd.o

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g
endif

all: test swimd_aligner

swimd_aligner: $(OBJECTS) swimd_aligner.cpp
	$(CC) $(CFLAGS) $(OBJECTS) swimd_aligner.cpp -o swimd_aligner

test: $(OBJECTS) test.cpp
	$(CC) $(CFLAGS) $(OBJECTS) test.cpp -o test

Swimd.o: Swimd.cpp Swimd.h
	$(CC) $(CFLAGS) -c Swimd.cpp

clean:
	rm *.o test swimd_aligner
