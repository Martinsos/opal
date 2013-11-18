CC = g++
CFLAGS = -msse4.2

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g
endif

all: test 

test: Swimd.o test.cpp
	$(CC) $(CFLAGS) test.cpp Swimd.o -o test

Swimd.o: Swimd.cpp Swimd.hpp
	$(CC) $(CFLAGS) -c Swimd.cpp

clean:
	rm *.o test
