CC = g++
LD = g++
LDFLAGS =
CFLAGS=-g -Wall `pkg-config --cflags libfreenect` `pkg-config --cflags opencv` -lopengl32 -lglut32 -msse3 -msse2
LIBS = `pkg-config --libs libfreenect` `pkg-config --libs opencv` -lGL -lGLU -lglut -lpthread
OBJECTS = main.o
PROG = test

all:$(PROG)

$(PROG): $(OBJECTS)
	$(LD) $(LDFLAGS) $(LIBS) $(OBJECTS) -o $(PROG)

%.o: %.cpp
	$(CC) $(CFLAGS)  $(LIBS) -c $<

clean:
	rm -rf *.o $(PROG)
