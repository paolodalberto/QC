# Makefile for rocBLAS ZGEMM Strided Batched Example


# Define the C++ compiler (CXX is a standard Make variable)
CXX = /opt/rocm/bin/hipcc

# Define the HIP compiler provided by ROCm
HIP_COMPILER = /opt/rocm/bin/hipcc
# ... rest of the file

# Define standard compiler flags for C++14/C++17 compatibility
HIP_FLAGS = -g -O1 --std=c++17 -I./

INCLUDE_DIRS = -I./

# Define linker flags to include rocBLAS and rocSOLVER libraries
# The -l prefix is standard for linking libraries
# Assuming libraries are in the default ROCm path (/opt/rocm/lib) which hipcc knows
LIBRARIES = -lrocblas 



# Source file
SOURCES = H.cpp H1.cpp circuit.cpp 

# Generate a list of required object files from the source files (e.g., davidson.o, sortarg.o, ...)
OBJECTS = $(SOURCES:.cpp=.o)


# Define the final executable name
EXECUTABLE = H


# Default target: builds the final executable
all: $(EXECUTABLE)

# Rule to link all object files and libraries into the final executable
# $^ automatically lists all prerequisites (all .o files)
$(EXECUTABLE): $(OBJECTS)
	@echo "------------------------------------------------------"
	@echo "Linking object files into the final executable: $@"
	$(HIP_COMPILER) $(HIP_FLAGS) $^ $(LIBRARIES) -o $@
	@echo "------------------------------------------------------"

# Pattern rule for building each object file from its corresponding source file
# The -c flag tells the compiler to stop after compilation and not link
%.o: %.cpp
	@echo "------------------------------------------------------"
	@echo "Compiling $< to object file $@"
	$(HIP_COMPILER) $(HIP_FLAGS)  $(INCLUDE_DIRS)  -c $< -o $@
	@echo "------------------------------------------------------"

# Clean target: removes compiled executables and object files
clean:
	@echo "Cleaning up build artifacts..."
	rm -f $(EXECUTABLE) $(OBJECTS)
	@echo "Cleanup complete."
