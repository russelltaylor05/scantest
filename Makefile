NVFLAGS=-g -O2 -arch=compute_20 -code=sm_20 
# list .c and .cu source files here
# use -02 for optimization during timed runs


SRCFILES=main.cu 
TARGET = ./scan

all:	scan

scan: $(SRCFILES) 
	nvcc $(NVFLAGS) -o scan $^

run: scan
	./scan  


double: $(SRCFILES)
	nvcc $(NVFLAGS) -DDOUBLE -o scan main.cu  

single: $(SRCFILES)
	nvcc $(NVFLAGS) -DSINGLE -o scan main.cu



test: $(TARGET)
	$(TARGET) input/A.in input/A.in

time: $(TARGET)
	time $(TARGET) input/1408.in input/1408.in


clean: 
	rm -f *.o scan