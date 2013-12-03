CC = g++
CFLAGS = -msse4.2 -O3
OBJECTS = Swimd.o

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g
endif

all: test 

test: $(OBJECTS) test.cpp
	$(CC) $(CFLAGS) $(OBJECTS) test.cpp -o test

Swimd.o: Swimd.cpp Swimd.h
	$(CC) $(CFLAGS) -c Swimd.cpp

clean:
	rm *.o test
