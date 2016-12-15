
NVCC		?= nvcc
NVCC_FLAGS	+= -ccbin g++ -m64 --std=c++11
NVCC_OPT_FLAGS 	?= -O2
NVCC_SM_FLAGS  	?= -gencode arch=compute_35,code=sm_35

SRC_FILES	= $(wildcard *.cu)
FILTERED_SRC    = $(filter-out main.cu,$(SRC_FILES))
FILTERED_OBJ    = $(FILTERED_SRC:%.cu=%.o)

TARGETS		= $(FILTERED_SRC:%.cu=%)

all : $(TARGETS)

rebuild : catch_main $(TARGETS)

%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OPT_FLAGS) $(NVCC_SM_FLAGS) $? -c


catch_runner : main.o 

% :  %.o catch_runner
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OPT_FLAGS) $(NVCC_SM_FLAGS) $< main.o -o $@ 

debug :
	@echo "NVCC           " $(NVCC)
	@echo "TARGETS        " $(TARGETS)
	@echo "NVCC_OPT_FLAGS " $(NVCC_OPT_FLAGS)
	@echo "NVCC_SM_FLAGS  " $(NVCC_SM_FLAGS) 

clean :
	-rm -f $(TARGETS) $(FILTERED_OBJ) 

.PHONY: help clean catch_runner 
.SECONDARY: $(FILTERED_OBJ)
