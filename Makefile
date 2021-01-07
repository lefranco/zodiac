
# requirements :
# sudo apt install python3-pip
# sudo pip3 install cython
# sudo apt-get install python3-dev


CYTHON = cython
CYTHONFLAGS = -3 --cplus --embed

CC = g++

OPT= -O3

# replace with proper python version here
CFLAGS = -fPIC $$(python3.8-config --cflags)
LDFLAGS = $$(python3.8-config --ldflags) -lpython3.8

PYTHON_SOURCES = homophonic.py
PYTHON_CPPS = $(PYTHON_SOURCES:.py=.cpp)
PYTHON_OBJS = $(PYTHON_CPPS:.cpp=.o)


all: homophonic

# py -> cpp
%.cpp: %.py 
	${CYTHON} ${CYTHONFLAGS}  $<

# cpp -> o
%.o: %.cpp
	${CC} -c ${OPT} ${CFLAGS} -o $@ $<

# o -> exe
homophonic: ${PYTHON_OBJS} 
	${CC} ${OPT} ${PYTHON_OBJS} ${LDFLAGS}  -o $@

clean:
	rm -f *.o
	rm -f main




