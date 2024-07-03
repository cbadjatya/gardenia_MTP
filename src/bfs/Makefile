include ../common.mk
# INCLUDES+=-I $(CUB_DIR)
# ICPCFLAGS+=-std=c++98
# B40_DIR=../../back40computing-read-only
# B40C_INC=-I $(B40_DIR) -I $(B40_DIR)/test
OBJS= main.o verifier.o 
all:  bfs_opt bfs_atomic_free
# bfs_linear_base bfs_linear_lb bfs_omp_beamer bfs_omp_base bfs_topo_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

ifeq ($(HOST),X86)
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
else
%.o: %.cc
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu
	/lfs/sware/cuda-11.2/bin/nvcc -g -G $(NVFLAGS) $(INCLUDES) -c $<

bfs_opt: bfs_opt.o $(OBJS)
	/lfs/sware/cuda-11.2/bin/nvcc -g -G $(NVFLAGS)  $(INCLUDES) $(EXTRA) $(OBJS)  bfs_opt.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_atomic_free: atomic_free.o $(OBJS)
	/lfs/sware/cuda-11.2/bin/nvcc -g -G $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS)  atomic_free.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
