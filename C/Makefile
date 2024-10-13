CC = gcc
CFLAGS = -Wall
TARGET = output_executable
SRCS = main.c arrith.c
OBJS = main.o arith.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

main.o: main.c arith.h
	$(CC) $(CFLAGS) -c main.c
arith.o: arith.c arith.h
	$(CC) $(CFLAGS) -c arith.c
clean:
	rm -f $(OBJS) $(TARGET)
