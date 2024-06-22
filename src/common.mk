CUDA_HOME=/lfs/sware/cuda-11.2
CUB_DIR=../../cub
OCL_DIR=/lfs/sware/cuda-11.2/targets/x86_64-linux/lib/
BIN=../../bin
HOST=X86
ifeq ($(HOST),X86)
CC=gcc
CXX=g++
else 
CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++
endif

NVCC=$(CUDA_HOME)/bin/nvcc
COMPUTECAPABILITY=sm_60
# CUDA_ARCH := \
# 	-gencode arch=compute_37,code=sm_37 \
# 	-gencode arch=compute_61,code=sm_61 \
# 	-gencode arch=compute_70,code=sm_70
CXXFLAGS=-Wall -fopenmp -std=c++11
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
#NVFLAGS+=-Xptxas -v
#NVFLAGS+=-cudart shared
ifeq ($(HOST),X86)
#SIMFLAGS=-O3 -Wall -DSIM -fopenmp -static -L/home/cxh/m5threads/ -lpthread
SIMFLAGS=-Wall -fopenmp -std=c++11 -O3 -g -static -lpthread -L$(GEM5_HOME)/m5threads
else
SIMFLAGS=-flto -fwhole-program -O3 -Wall -fopenmp -static -g
endif
#M5OP=$(GEM5_HOME)/util/m5/src/arm/m5op.S
M5OP=$(GEM5_HOME)/util/m5/src/x86/m5op.S
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G -lineinfo
else
	CXXFLAGS += -g -O3
	NVFLAGS += -O3 -w
endif
# CU_INC = -I/usr/include/cuda
CU_INC = -I$(CUDA_HOME)/include
INCLUDES = -I../../include
#INCLUDES += $(CU_INC)
LIBS = -L$(CUDA_HOME)/lib64
# LIBS = -L/usr/lib64



VPATH += ../common
OBJS=main.o VertexSet.o graph.o
