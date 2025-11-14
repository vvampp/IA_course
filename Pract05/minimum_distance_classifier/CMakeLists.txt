.PHONY: all build test clean help install python-module cuda-test cpu-test

# Default target
all: build

# Build everything
build:
	@echo "Building project..."
	@./build_and_test.sh --no-tests

# Build and run tests
test:
	@echo "Building and testing..."
	@./build_and_test.sh

# Build only CPU version
cpu:
	@echo "Building CPU-only version..."
	@./build_and_test.sh --no-cuda

# Run only CUDA tests
cuda-test:
	@echo "Running CUDA tests..."
	@cd build && ./test_classifier --gtest_filter="*CUDA*"

# Run only CPU tests
cpu-test:
	@echo "Running CPU tests..."
	@cd build && ./test_classifier --gtest_filter="-*CUDA*"

# Clean build artifacts
clean:
	@echo "Cleaning build directory..."
	@rm -rf build
	@echo "Clean complete!"

# Clean and rebuild
rebuild: clean build

# Run tests with verbose output
test-verbose:
	@cd build && ./test_classifier --gtest_print_time=1

# Run specific test suite
test-basic:
	@cd build && ./test_classifier --gtest_filter="MinimumDistanceClassifierTest.*"

test-validation:
	@cd build && ./test_classifier --gtest_filter="*Validation*"

test-accuracy:
	@cd build && ./test_classifier --gtest_filter="*Accuracy*"

# List all tests
list-tests:
	@cd build && ./test_classifier --gtest_list_tests

# Install Python module (development mode)
install-dev:
	@echo "Installing Python module in development mode..."
	@pip install -e python/

# Show help
help:
	@echo "Minimum Distance Classifier - Makefile Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build project (default)"
	@echo "  build          - Build project without tests"
	@echo "  test           - Build and run all tests"
	@echo "  cpu            - Build CPU-only version"
	@echo "  cuda-test      - Run only CUDA tests"
	@echo "  cpu-test       - Run only CPU tests"
	@echo "  clean          - Remove build directory"
	@echo "  rebuild        - Clean and rebuild"
	@echo "  test-verbose   - Run tests with timing information"
	@echo "  test-basic     - Run basic functionality tests"
	@echo "  test-validation- Run validation tests"
	@echo "  test-accuracy  - Run accuracy tests"
	@echo "  list-tests     - List all available tests"
	@echo "  install-dev    - Install Python module (dev mode)"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build project"
	@echo "  make test         # Build and run all tests"
	@echo "  make clean test   # Clean rebuild and test"
	@echo "  make cpu test     # Build CPU version and test"
