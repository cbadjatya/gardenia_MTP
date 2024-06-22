include ../common.mk
SRC=main.cc
OBJS=main.o
INCLUDES += -I${MKLROOT}/include
# LIBS += -L${MKLROOT}/lib/intel64
all: $(OBJS) spmv_base swell_32 spmv_opt

%.o: %.cpp
	$(ICPC) $(INCLUDES) -c $<

ifeq ($(HOST),X86)
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
else
%.o: %.cc
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu
	$(NVCC)  $(INCLUDES) -c $<


spmv_base: main.o base.o
	$(NVCC)   $(INCLUDES) main.o base.o -o $@ 
	mv $@ $(BIN)

swell_32: main.o swell_32.o
	$(NVCC)   $(INCLUDES) main.o swell_32.o -o $@ 
	mv $@ $(BIN)

swell_64: main.o swell_64.o
	$(NVCC)   $(INCLUDES) main.o swell_64.o -o $@ 
	mv $@ $(BIN)

swell_1024: main.o swell_1024.o
	$(NVCC)   $(INCLUDES) main.o swell_1024.o -o $@ 
	mv $@ $(BIN)


spmv_opt: main.o opt.o
	$(NVCC)  $(INCLUDES) main.o opt.o -o $@ $(LIBS)
	mv $@ $(BIN)


clean:
	rm *.o
