CXX = g++
GCCVERSION = $(shell gcc --version | grep ^gcc | sed 's/^.* //g')

ifeq "$(GCCVERSION)" "4.9.1"
    CXX = g++-4.8
endif

CUDA_PATH=/usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)  
OS_SIZE    = $(shell uname -m | sed -e "s/x86_64/64/" -e "s/armv7l/32/" -e "s/aarch64/64/")
OS_ARCH    = $(shell uname -m)
ARCH_FLAGS =
# internal flags
NVCCFLAGS   := -m${OS_SIZE} ${ARCH_FLAGS}
CCFLAGS     :=
LDFLAGS     :=

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?=

LDFLAGS += --sysroot=$(TARGET_FS)
LDFLAGS += -rpath-link=$(TARGET_FS)/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-$(abi)

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      TARGET := debug
else
      TARGET := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))
ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM32    := -gencode arch=compute_32,code=sm_32
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM50    := -gencode arch=compute_50,code=sm_50
GENCODE_SMXX    := -gencode arch=compute_50,code=compute_50
GENCODE_FLAGS   ?= $(GENCODE_SM10) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM32) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SMXX)

SRC_DIR = src
OBJ_DIR = obj

CFLAGS = -I$(SRC_DIR) -I$(CUDA_PATH)/include  -std=c++11
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart 


SRC = $(shell find $(SRC_DIR)/ -type f -name '*.cpp' -o -name '*.cu')
OBJ := $(SRC:.cu=.o)
OBJ := $(OBJ:.cpp=.o)
OBJ := $(subst $(SRC_DIR), $(OBJ_DIR), $(OBJ))


all: build

build: chary

test: clean  all
	@echo "Running chary"
	@./chary data/queryfile data/dbfile

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $
	$(CXX) $(CFLAGS)  -o  $@  -c $?

chary: $(OBJ)
	$(CXX) $? -L$(CUDA_PATH)/lib64 -lcudart -lcuda    -o $@

$(OBJ): | $(OBJ_DIR)


$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

clean:
	@echo "Removing object files, dok files and executable file"
	rm -rf $(OBJ_DIR) chary dok.*

doc:
	pdflatex documentation/dok.tex 

clobber: clean

