
NVCC		?= nvcc
NVCC_FLAGS	+= -ccbin g++ -m64 
NVCC_OPT_FLAGS 	?= -O2
NVCC_SM_FLAGS  	?= -gencode arch=compute_35,code=sm_35

SRC_FILES	= $(wildcard *.cu)
TARGETS		= $(SRC_FILES:%.cu=%)

all : $(TARGETS)

% : %.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_OPT_FLAGS) $(NVCC_SM_FLAGS) $< -o $@ 

help :
	@echo "NVCC           " $(NVCC)
	@echo "TARGETS        " $(TARGETS)
	@echo "NVCC_OPT_FLAGS " $(NVCC_OPT_FLAGS)
	@echo "NVCC_SM_FLAGS  " $(NVCC_SM_FLAGS) 
