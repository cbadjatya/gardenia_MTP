include ../common.mk
CXXFLAGS+=-std=c++11
SRC=main.cc
OBJS=verifier.o main.o

all: $(OBJS) symgs_base symgs_opt

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) -g -G $(INCLUDES) -c $<

symgs_omp_base: omp_base.o main.o verifier.o ../vc/omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o ../vc/omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

symgs_omp_target: main.o omp_target.o verifier.o ../vc/omp_base.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o ../vc/omp_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

symgs_base: main.o base.o verifier.o ../vc/omp_base.o
	$(NVCC) -g -G $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o ../vc/omp_base.o -o $@ $(LIBS) -lgomp
	mv $@ $(BIN)

symgs_opt: main.o opt.o verifier.o ../vc/omp_base.o
	$(NVCC) -g -G $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o opt.o verifier.o ../vc/omp_base.o -o $@ $(LIBS) -lgomp
	mv $@ $(BIN)

symgs_tex: main.o tex.o verifier.o ../vc/omp_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o tex.o verifier.o ../vc/omp_base.o -o $@ $(LIBS) -lgomp
	mv $@ $(BIN)

symgs_warp: main.o warp.o verifier.o ../vc/omp_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o warp.o verifier.o ../vc/omp_base.o -o $@ $(LIBS) -lgomp
	mv $@ $(BIN)

symgs_vector: main.o vector.o verifier.o ../vc/omp_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o vector.o verifier.o ../vc/omp_base.o -o $@ $(LIBS) -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
