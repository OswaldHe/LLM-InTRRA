GCC=g++
PLATFORM := xilinx_u280_xdma_201920_3
TOP_NAME := gating_net
CONNECTIVITY_FILE := hbm_config_sequential.ini

CPU_EXECUTABLE := build/gating-net
SEQUENTIAL_EXECUTABLE := build/gating-net-sequential

CPU_SRCS := ../gating-net.cpp
SEQUENTIAL_HOST_SRCS := gating-net-sequential-host.cpp
SEQUENTIAL_KERNEL_SRCS := gating-net-sequential-kernel.cpp
BASELINE_SRCS := $(SEQUENTIAL_KERNEL_SRCS) $(SEQUENTIAL_HOST_SRCS)

DUMMY := $(shell mkdir -p build)

$(CPU_EXECUTABLE): $(CPU_SRCS)
	$(GCC) -o $(CPU_EXECUTABLE) -O2 $^ -lm

$(SEQUENTIAL_EXECUTABLE): $(BASELINE_SRCS)
	$(GCC) -o $(SEQUENTIAL_EXECUTABLE) -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include


.PHONY: build_sequential
build_sequential: $(SEQUENTIAL_KERNEL_SRCS)
	rm -rf work.out/ build/
	mkdir -p build
	tapac \
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

.PHONY: build_sequential_hw_emu
build_sequential_hw_emu:
	cd work.out/run-1/;\
	sed -i '2s/.*/# TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/TARGET=hw_emu/' generate-bitstream.sh; \
	sed -i '4s/.*/DEBUG=-g/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw_emu/gating_net_$(PLATFORM).xclbin bitstreams/gating_net_sequential_$(PLATFORM)_hw_emu.xclbin

.PHONY: build_sequential_hw
build_sequential_hw:
	cd work.out/run-1/;\
	sed -i '2s/.*/TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/# TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw/gating_net_$(PLATFORM).xclbin bitstreams/gating_net_sequential_$(PLATFORM)_hw.xclbin


gating-net: gating-net.cpp
	$(GCC) -o $@ -O2 $^ -lm

gating-net-sequential: gating-net-sequential-kernel.cpp gating-net-sequential-host.cpp
	$(GCC) -o $@ -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

run-gating-net-sequential: gating-net-sequential
	./gating-net-sequential

# Run the designs
run_cpu: $(CPU_EXECUTABLE)
	./$(CPU_EXECUTABLE)

run_sequential: $(SEQUENTIAL_EXECUTABLE)
	./$(SEQUENTIAL_EXECUTABLE)

run_sequential_hw_emu:
	./$(SEQUENTIAL_EXECUTABLE) --bitstream bitstreams/gating_net_sequential_$(PLATFORM)_hw_emu.xclbin

run_sequential_hw:
	./$(SEQUENTIAL_EXECUTABLE) --bitstream bitstreams/gating_net_sequential_$(PLATFORM)_hw.xclbin

clean:
	rm -rf gating-net gating-net-sequential work.out/ build/