include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cc
#OBJS=omp_base.o topo_base.o topo_bitset.o topo_warp.o linear_base.o linear_bitset.o linear_warp.o main.o verifier.o
OBJS=main.o verifier.o topo_base.o topo_opt.o

all: $(OBJS) vc_topo_base vc_topo_opt vc_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) -g -G $(NVFLAGS) $(INCLUDES) -c $<

vc_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

vc_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

vc_topo_base: main.o topo_base.o verifier.o
	$(NVCC) -g -G $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_base.o verifier.o -o $@
	mv $@ $(BIN)

vc_topo_opt: main.o topo_opt.o verifier.o
	$(NVCC) -g -G $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_opt.o verifier.o -o $@
	mv $@ $(BIN)


vc_topo_bitset: main.o topo_bitset.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_bitset.o verifier.o -o $@
	mv $@ $(BIN)

vc_topo_warp: main.o topo_warp.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_warp.o verifier.o -o $@
	mv $@ $(BIN)

vc_linear_base: main.o linear_base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_base.o verifier.o -o $@
	mv $@ $(BIN)

vc_linear_bitset: main.o linear_bitset.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_bitset.o verifier.o -o $@
	mv $@ $(BIN)

vc_linear_warp: main.o linear_warp.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_warp.o verifier.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o

