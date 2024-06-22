include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu
EXE=mst

all: $(EXE)_main

$(EXE)_main:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -o $@
	mv $@ $(BIN)

clean:
	rm *.o

