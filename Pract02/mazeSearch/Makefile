# compiler
CC = g++

# compiler flags
CFLAGS = -std=c++17 -Wall

# SFML libraries
SFML_LIBRARIES = -lsfml-graphics -lsfml-window -lsfml-system -lsfml-audio

SOURCE = *.cpp

# Build target
TARGET = mazes

all: $(TARGET)
	@echo "\n Build completed successfully! Run ./$(TARGET) to execute the program!"

$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(CFLAGS) $(SFML_INCLUDE) $(SFML_LIBRARIES)

clean:
	rm -f $(TARGET)
