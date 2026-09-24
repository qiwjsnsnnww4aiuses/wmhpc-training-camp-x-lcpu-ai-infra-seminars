NVCC ?= nvcc
ARCH ?= 103a
FLAGS ?= -O3 -std=c++17 -lineinfo --expt-relaxed-constexpr
GENCODE = -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

TARGET = bin/instruction_only_mma
SOURCE = instruction_only_mma.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	@mkdir -p $(dir $@)
	$(NVCC) $(FLAGS) $(GENCODE) -o $@ $<

run: $(TARGET)
	./$(TARGET)

sass: $(TARGET)
	cuobjdump --dump-sass $(TARGET) > bin/instruction_only_mma.sass
	@echo "wrote bin/instruction_only_mma.sass"

clean:
	rm -f $(TARGET) bin/instruction_only_mma.sass

.PHONY: all run sass clean
