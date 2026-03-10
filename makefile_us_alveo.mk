#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3
#

############################## Help Section ##############################
ifneq ($(findstring Makefile, $(MAKEFILE_LIST)), Makefile)
help:
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Build host + xclbin for the given TARGET."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Run the application (emulation or hardware)."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform>"
	$(ECHO) "      Build only xclbin."
	$(ECHO) ""
	$(ECHO) "  make host"
	$(ECHO) "      Build only host executable."
	$(ECHO) ""
	$(ECHO) "  make clean / cleanall"
	$(ECHO) ""
endif

############################## Setting up Project Variables ##############################
# IMPORTANT: must be overridable from command line
TARGET ?= hw
include ./utils.mk

TEMP_DIR   := ./_x.$(TARGET).$(XSA)
BUILD_DIR  := ./build_dir.$(TARGET).$(XSA)
EMCONFIG_DIR := $(TEMP_DIR)

PACKAGE_OUT := ./package.$(TARGET)

# If the top Makefile exports these, we use them; otherwise give safe defaults
KERNEL_NAME ?= knn
KERNEL_SRC  ?= knn.cpp
HOST_SRC    ?= host.cpp
EXE_NAME    ?= knn
XCLBIN_NAME ?= knn.xclbin

EXECUTABLE := ./$(EXE_NAME)
XCLBIN_OUT := $(BUILD_DIR)/$(XCLBIN_NAME)
LINK_OUTPUT := $(BUILD_DIR)/$(EXE_NAME).link.xclbin
CMD_ARGS   := $(XCLBIN_OUT)

############################## Tool flags ##############################
VPP_PFLAGS :=
VPP_FLAGS  += -t $(TARGET) --platform $(PLATFORM) --save-temps

CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -I/usr/include/x86_64-linux-gnu -Wall -O0 -g -std=c++1y
LDFLAGS  += -L$(XILINX_XRT)/lib -pthread -lOpenCL

########################## Checking if PLATFORM in allowlist #######################
PLATFORM_BLOCKLIST += nodma

############################## Setting up Host Variables ##############################
CXXFLAGS   += -I$(XF_PROJ_ROOT)/common/includes/xcl2
HOST_SRCS  += $(XF_PROJ_ROOT)/common/includes/xcl2/xcl2.cpp ./$(HOST_SRC)

CXXFLAGS   += -fmessage-length=0
LDFLAGS    += -lrt -lstdc++

############################## Optional: pass dataset size to both host and kernel ##############################
# Usage: make all TARGET=hw_emu NUM_PT_IN_SEARCHSPACE=512
ifneq ($(NUM_PT_IN_SEARCHSPACE),)
CXXFLAGS  += -DNUM_PT_IN_SEARCHSPACE=$(NUM_PT_IN_SEARCHSPACE)
VPP_FLAGS += -DNUM_PT_IN_SEARCHSPACE=$(NUM_PT_IN_SEARCHSPACE)
endif

############################## Targets ##############################
.PHONY: all host build xclbin run test emconfig clean cleanall

# Build only. Do NOT auto-run inside "all".
all: check-platform check-device check-vitis $(EXECUTABLE) $(XCLBIN_OUT) emconfig

#host: check-xrt $(EXECUTABLE)

build: check-vitis check-device $(XCLBIN_OUT)

xclbin: build

############################## Kernel build rules ##############################
$(TEMP_DIR)/kernel.xo: $(KERNEL_SRC)
	mkdir -p $(TEMP_DIR)
	v++ $(VPP_FLAGS) -c -k $(KERNEL_NAME) --temp_dir $(TEMP_DIR) -I'$(<D)' -o'$@' '$<'

$(XCLBIN_OUT): $(TEMP_DIR)/kernel.xo
	mkdir -p $(BUILD_DIR)
	v++ $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(TEMP_DIR) -o'$(LINK_OUTPUT)' $(+)
	v++ -p $(LINK_OUTPUT) $(VPP_FLAGS) --package.out_dir $(PACKAGE_OUT) -o $(XCLBIN_OUT)

############################## Host build rule ##############################
$(EXECUTABLE): $(HOST_SRCS)
	g++ -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

############################## Emulation config ##############################
emconfig: $(EMCONFIG_DIR)/emconfig.json

$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(EMCONFIG_DIR)

############################## Run rules ##############################
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	cp -rf $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(EXECUTABLE) $(CMD_ARGS)
endif

test: run

############################## Cleaning rules ##############################
clean:
	-$(RMDIR) $(EXECUTABLE)
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	-$(RMDIR) *.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb

cleanall: clean
	-$(RMDIR) build_dir*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin
