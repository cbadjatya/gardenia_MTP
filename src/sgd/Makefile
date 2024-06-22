include ../common.mk
INCLUDES+=-I$(CUB_DIR)
SRC=main.cc
OBJS=base.o main.o verifier.o
CXXFLAGS+=-std=c++11

all: $(OBJS) sgd_base sgd_opt

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cc sgd.h
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

sgd_base: base.o main.o verifier.o sgd.h
	$(NVCC) $(NVFLAGS) -g -G $(INCLUDES) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgd_opt: opt.o main.o verifier.o sgd.h
	$(NVCC) $(NVFLAGS) -g -G $(INCLUDES) main.o opt.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
