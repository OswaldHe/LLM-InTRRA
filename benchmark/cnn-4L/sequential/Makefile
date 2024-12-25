GCC := g++
PLATFORM := xilinx_u280_xdma_201920_3
TOP_NAME := CNN4L
CONNECTIVITY_FILE := hbm_config.ini

CPU_EXECUTABLE := build/cnn-4L
HLS_EXECUTABLE := build/cnn-4L-spatial

CPU_SRCS := cnn-4L.cpp

HLS_HOST_SRCS := cnn-4L-spatial-host.cpp
HLS_KERNEL_SRCS := cnn-4L-sequential-kernel.cpp
HLS_SRCS := $(HLS_KERNEL_SRCS) $(HLS_HOST_SRCS)

DUMMY := $(shell mkdir -p build)

$(CPU_EXECUTABLE): $(CPU_SRCS)
	$(GCC) -o $(CPU_EXECUTABLE) -O2 $^ -lm

$(HLS_EXECUTABLE): $(HLS_SRCS)
	$(GCC) -o $(HLS_EXECUTABLE) -O2 $^ -L/lib/x86_64-linux-gnu -L/usr/local/lib -ltapa -lfrt -lgflags -lglog -lm -lOpenCL -I$(XILINX_HLS)/include

.PHONY: build_tapa
build_tapa: $(HLS_KERNEL_SRCS)
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

.PHONY: build_hw_emu
build_hw_emu:
	cd work.out/run-1/;\
	sed -i '2s/.*/# TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw_emu/CNN4L_$(PLATFORM).xclbin bitstreams/CNN4L_$(PLATFORM)_hw_emu.xclbin

.PHONY: build_hw
build_hw:
	cd work.out/run-1/;\
	sed -i '2s/.*/TARGET=hw/' generate-bitstream.sh; \
	sed -i '3s/.*/# TARGET=hw_emu/' generate-bitstream.sh; \
	bash generate-bitstream.sh
	mkdir -p bitstreams/; \
	cp work.out/run-1/vitis_run_hw/CNN4L_$(PLATFORM).xclbin bitstreams/CNN4L_$(PLATFORM)_hw.xclbin

run_cpu: $(CPU_EXECUTABLE)
	./$(CPU_EXECUTABLE)

run_hls: $(HLS_EXECUTABLE)
	./$(HLS_EXECUTABLE)

run_hw_emu:
	./$(HLS_EXECUTABLE) --bitstream bitstreams/CNN4L_$(PLATFORM)_hw_emu.xclbin

run_hw:
	./$(HLS_EXECUTABLE) --bitstream bitstreams/CNN4L_$(PLATFORM)_hw.xclbin

.PHONY: build_hw
clean:
	rm -rf work.out/ build/
