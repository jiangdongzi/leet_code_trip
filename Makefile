CXX ?= g++

BUILD_DIR := build
TARGET := hello

SRC := main.cpp
OBJ := $(BUILD_DIR)/main.o

# Debug-friendly build: minimal optimization and full debug info.
CXXFLAGS ?= -std=c++11 -Og -g -Wall -Wextra -Wpedantic

FMT_CFLAGS := $(shell pkg-config --cflags fmt 2>/dev/null)
FMT_LIBS := $(shell pkg-config --libs fmt 2>/dev/null)

CPPFLAGS += $(FMT_CFLAGS)
CPPFLAGS += -MMD -MP
LDLIBS += $(if $(strip $(FMT_LIBS)),$(FMT_LIBS),-lfmt)

.PHONY: all run gdb clean
.PHONY: format

all: $(TARGET)

DEP := $(OBJ:.o=.d)
-include $(DEP)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/main.o: $(SRC) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

gdb: $(TARGET)
	gdb -q ./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

format:
	clang-format -i $$(find . -path ./build -prune -o -path ./.git -prune -o \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.h' -o -name '*.hpp' \) -print)
