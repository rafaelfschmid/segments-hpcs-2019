#TIME=1
#BITS=12
#EXECS=1
#STREAMS=32

arch=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60

all: mergeseg radixseg fixcub fixthrust nthrust fixseq mseq

mergeseg: mergeseg.cu
	nvcc $(arch) mergeseg.cu -o mergeseg.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

radixseg: radixseg.cu
	nvcc $(arch) radixseg.cu -o radixseg.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixcub: fixcub.cu
	nvcc $(arch) fixcub.cu -o fixcub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
 
fixthrust: fixthrust.cu
	nvcc $(arch) fixthrust.cu -o fixthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixpasscub: fixpasscub.cu
	nvcc $(arch) fixpasscub.cu -o fixpasscub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
 
fixpassthrust: fixpassthrust.cu
	nvcc $(arch) fixpassthrust.cu -o fixpassthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

nthrust: nthrust.cu
	nvcc $(arch) nthrust.cu -o nthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixseq: fixseq.cu
	nvcc $(arch) fixseq.cu -o fixseq.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixpassseq: fixpassseq.cu
	nvcc $(arch) fixpassseq.cu -o fixpassseq.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

mseq: mseq.cu
	nvcc $(arch) mseq.cu -o mseq.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

ncub: ncub.cu
	nvcc $(arch) ncub.cu -o ncub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

nmoderngpu: nmoderngpu.cu
	nvcc $(arch) nmoderngpu.cu -o nmoderngpu.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)


