CXX=g++
#CXXFLAGS=-fopenmp -static -O3
CXXFLAGS=-fopenmp -fPIC -pipe  -O3
#CXXFLAGS = -fPIC
VERSION=1.41

all: omp-pmf-train omp-pmf-predict

omp-pmf-train: pmf-train.cpp pmf.h util.o ccd-r1.o pcr.o
	${CXX} ${CXXFLAGS} -o omp-pmf-train pmf-train.cpp ccd-r1.o util.o pcr.o

omp-pmf-predict: pmf-predict.cpp pmf.h util.o pcr.o
	${CXX} ${CXXFLAGS} -o omp-pmf-predict pmf-predict.cpp  util.o pcr.o

pcr.o: pcr.cpp util.o
	${CXX} ${CXXFLAGS} -c -o pcr.o pcr.cpp 

ccd-r1.o: ccd-r1.cpp util.o
	${CXX} ${CXXFLAGS} -c -o ccd-r1.o ccd-r1.cpp 

util.o: util.h util.cpp
	${CXX} ${CXXFLAGS} -c -o util.o util.cpp

tar: 
	make clean; cd ../;  tar cvzf libpmf-${VERSION}.tgz libpmf-${VERSION}/

clean:
	make -C python clean
	make -C R clean
	make -C matlab
	rm -rf  omp-pmf* *.o 

