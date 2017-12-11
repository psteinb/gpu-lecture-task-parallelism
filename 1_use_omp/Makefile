
NVCC		?= nvcc
NVCC_FLAGS	+= -ccbin g++ -m64 --std=c++11 -I. -I..
CXXFLAGS 	?= -Xcompiler -fopenmp -O2
SMS  	?= -gencode arch=compute_35,code=sm_35

SRC_FILES	= $(wildcard *.cu)
FILTERED_SRC    = $(filter-out main.cu,$(SRC_FILES))
FILTERED_OBJ    = $(FILTERED_SRC:%.cu=%.o)

TARGETS		= $(FILTERED_SRC:%.cu=%)

all : $(TARGETS)

rebuild : catch_main $(TARGETS)

%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) $(CXXFLAGS) $(SMS) $? -c


catch_runner : main.o 

% :  %.o catch_runner
	$(NVCC) $(NVCC_FLAGS) $(CXXFLAGS) $(SMS) $< main.o -o $@

debug :
	@echo "NVCC           " $(NVCC)
	@echo "TARGETS        " $(TARGETS)
	@echo "CXXFLAGS       " $(CXXFLAGS)
	@echo "NVCC_SM_FLAGS  " $(SMS)

clean :
	-rm -f $(TARGETS) $(FILTERED_OBJ) 

.PHONY: help clean catch_runner 
.SECONDARY: $(FILTERED_OBJ)
