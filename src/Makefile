CC = g++
CFLAGS = -msse4.2
OBJECTS = Swimd.o $(foreach p,8 16 32,Swimd_searchDatabase$(p).o)

DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS += -g
endif

all: test 

test: $(OBJECTS) test.cpp
	$(CC) $(CFLAGS) $(OBJECTS) test.cpp -o test

Swimd.o: Swimd.cpp Swimd.h
	$(CC) $(CFLAGS) -c Swimd.cpp

# I generate multiple .o files from Swimd_searchDatabase.cpp,
# each containing database search function for specific precision.
Swimd_searchDatabase%.o: Swimd_searchDatabase.cpp Swimd.h
	$(CC) $(CFLAGS) -DSSE_$* -c Swimd_searchDatabase.cpp -o $@

clean:
	rm *.o test
