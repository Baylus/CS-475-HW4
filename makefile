BIN  = nn

CXX=g++
CC=g++
SHELL=/bin/sh

CPPFLAGS= -L /usr/lib/libp/  -pg
GCPPFLAGS=-g -Wall
CPPFLAGS=-O3 -Wall

CFLAGS=$(CPPFLAGS)
LIBS = -lm

SRCS=\
nn.cpp\
mat.cpp\
rand.cpp\
randf.cpp\
randmt.cpp

HDRS=\
mat.h\
rand.h

OBJS=\
mat.o\
rand.o

$(BIN):	$(OBJS) $(BIN).o
	$(CC) $(CFLAGS) $(OBJS) $(BIN).o $(LIBS) -o $(BIN)

$(BIN)oneof:	$(OBJS) $(BIN)oneof.o
	$(CC) $(CFLAGS) $(OBJS) $(BIN)oneof.o $(LIBS) -o $(BIN)oneof

debug$(BIN):	$(OBJS) $(BIN).o
	$(CC) $(GCPPFLAGS) $(OBJS) $(BIN).o $(LIBS) -o $(BIN)

debug$(BIN)oneof:	$(OBJS) $(BIN)oneof.o
	$(CC) $(GCPPFLAGS) $(OBJS) $(BIN)oneof.o $(LIBS) -o $(BIN)oneof

size:	$(HDRS)  $(SRCS) 
	wc -l $?

srcs:	$(HDRS)  $(SRCS) 
	echo $(HDRS)  $(SRCS) 

all:
	touch $(HDRS)  $(SRCS) 

debug:	$(OBJS) nn.cpp
	$(CC) $(GCPPFLAGS) $(OBJS) nn.cpp $(LIBS) -o nn

clean:
	/bin/rm -f *.o $(BIN)*.tar *~ core gmon.out a.out

tar: makefile $(SRCS) $(HDRS)
	tar -cvzf "$(date +"%Y-%m-%d") $(BIN).tar" $(SRCS) $(HDRS) $(DOCS)
	ls -l $(BIN)*tar