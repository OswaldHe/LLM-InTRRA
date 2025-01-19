GCC=g++
PLATFORM := xilinx_u280_xdma_201920_3
TOP_NAME := gating_net
CONNECTIVITY_FILE := hbm_config_spatial.ini

CPU_EXECUTABLE := build/gating-net
SPATIAL_EXECUTABLE := build/gating-net-spatial

CPU_SRCS := ../gating-net.cpp
SPATIAL_HOST_SRCS := gating-net-spatial-host.cpp
SPATIAL_KERNEL_SRCS := gating-net-spatial-kernel.cpp
BASELINE_SRCS := $(SPATIAL_KERNEL_SRCS) $(SPATIAL_HOST_SRCS)

DUMMY := $(shell mkdir -p build)

$(CPU_EXECUTABLE): $(CPU_SRCS)
	$(GCC) -o $(CPU_EXECUTABLE) -O2 $^ -lm

$(SPATIAL_EXECUTABLE): $(BASELINE_SRCS)
	$(GCC) -o $(SPATIAL_EXECUTABLE) -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include


.PHONY: build_spatial
build_spatial: $(SPATIAL_KERNEL_SRCS)
	rm -rf work.out/ build/
	mkdir -p build
	tapac \
		-v 10 \
		-o build/solver.$(PLATFORM).hw.xo \
		--platform $(PLATFORM) \
		--top $(TOP_NAME) \
		--work-dir build/solver.$(PLATFORM).hw.xo.tapa \
		--connectivity $(CONNECTIVITY_FILE) \
		--enable-hbm-binding-adjustment \
		--enable-synth-util \
		--run-floorplan-dse \
		--max-parallel-synth-jobs 16 \
		--floorplan-output build/solver.tcl \
		$^

.PHONY: build_spatial_hw_emu
build_spatial_hw_emu:
	cd work.out/run-1/;\
	sed -i '2s/.*/# TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/TARGET=hw_emu/' generate-bitstream.sh; \
	sed -i '4s/.*/DEBUG=-g/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw_emu/gating_net_$(PLATFORM).xclbin bitstreams/gating_net_spatial_$(PLATFORM)_hw_emu.xclbin

.PHONY: build_spatial_hw
build_spatial_hw:
	cd work.out/run-1/;\
	sed -i '2s/.*/TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/# TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw/gating_net_$(PLATFORM).xclbin bitstreams/gating_net_spatial_$(PLATFORM)_hw.xclbin


gating-net: gating-net-spatial-host.cpp gating-net-spatial-kernel.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

gating-net-spatial: gating-net-spatial-kernel.cpp gating-net-spatial-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

run-gating-net: gating-net-spatial
	./gating-net-spatial 2> ~/scratch/spatial_csim.log
	cp ~/scratch/spatial_csim.log .

# Run the designs
run_cpu: $(CPU_EXECUTABLE)
	./$(CPU_EXECUTABLE)

run_spatial: $(SPATIAL_EXECUTABLE)
	./$(SPATIAL_EXECUTABLE)

run_spatial_hw_emu:
	./$(SPATIAL_EXECUTABLE) --bitstream bitstreams/gating_net_spatial_$(PLATFORM)_hw_emu.xclbin

run_spatial_hw:
	./$(SPATIAL_EXECUTABLE) --bitstream bitstreams/gating_net_spatial_$(PLATFORM)_hw.xclbin

clean:
	rm -rf gating-net gating-net-spatial work.out/ build/