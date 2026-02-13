# Duckt Language Makefile
# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
DEBUG_FLAGS = -g -O0 -DDEBUG

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/lexer.cpp \
          $(SRC_DIR)/parser.cpp \
          $(SRC_DIR)/interpreter.cpp \
          $(SRC_DIR)/main.cpp

# Object files
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Target executable
TARGET = $(BIN_DIR)/duckt

# Default target
all: directories $(TARGET)

# Create directories
directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean all

# Run
run: all
	./$(TARGET)

# Run with file
run-file: all
	@if [ -z "$(FILE)" ]; then echo "Usage: make run-file FILE=<path>"; exit 1; fi
	./$(TARGET) $(FILE)

# Run examples
examples: all
	@echo "=== Running hello.dt ==="
	./$(TARGET) examples/hello.dt
	@echo ""
	@echo "=== Running tensors.dt ==="
	./$(TARGET) examples/tensors.dt
	@echo ""
	@echo "=== Running neural_net.dt ==="
	./$(TARGET) examples/neural_net.dt
	@echo ""
	@echo "=== Running agent.dt ==="
	./$(TARGET) examples/agent.dt

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Install (to /usr/local/bin)
install: all
	cp $(TARGET) /usr/local/bin/duckt

# Uninstall
uninstall:
	rm -f /usr/local/bin/duckt

# Help
help:
	@echo "Duckt Language Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all       Build the Duckt interpreter (default)"
	@echo "  debug     Build with debug symbols"
	@echo "  run       Build and start the REPL"
	@echo "  run-file  Build and run a file (FILE=path)"
	@echo "  examples  Run all example programs"
	@echo "  clean     Remove build artifacts"
	@echo "  install   Install to /usr/local/bin"
	@echo "  uninstall Remove from /usr/local/bin"
	@echo "  help      Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make                  # Build"
	@echo "  make run              # Start REPL"
	@echo "  make run-file FILE=examples/hello.dt"
	@echo "  make debug            # Debug build"

.PHONY: all directories debug run run-file examples clean install uninstall help
