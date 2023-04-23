.phony: clean default

BINDIR=./bin

default: $(BINDIR) run_omp_hello run_omp_ver

$(BINDIR):
	mkdir -p $(BINDIR)

run_omp_hello: omp_hello.c 
	g++ -g -Wall -fopenmp -o ${BINDIR}/$@ $^

run_omp_ver: omp_ver.cpp
	g++ -std=c++11 -fopenmp -o ${BINDIR}/$@ omp_ver.cpp


clean: ${BINDIR}
	rm -rf $^