all: vector_add run

# Compile Metal shader to .air
Shaders.air: add.metal
	xcrun -sdk macosx metal -c add.metal -o Shaders.air

# Compile .air to .metallib
default.metallib: Shaders.air
	xcrun -sdk macosx metallib Shaders.air -o default.metallib

# Compile C++ code
vector_add: main.mm default.metallib
	clang++ -framework Metal -framework Foundation -o vector_add main.mm

# Run the program
run: vector_add
	./vector_add

# Clean up build artifacts
clean:
	rm -f Shaders.air default.metallib vector_add