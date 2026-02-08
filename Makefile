CXX ?= g++

BUILD_DIR ?= build
TARGET ?= hello

SRC := main.cpp
OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC))

# Debug-friendly build: minimal optimization and full debug info.
CXXSTD ?= -std=c++11
# Default to quiet builds; override e.g. `make CXXWARN='-Wall -Wextra -Wpedantic'`.
CXXWARN ?= -w
CXXFLAGS ?= $(CXXSTD) -Og -g $(CXXWARN)

ASAN_CXXFLAGS := -fsanitize=address -fno-omit-frame-pointer
ASAN_LDFLAGS := -fsanitize=address

FMT_CFLAGS := $(shell pkg-config --cflags fmt 2>/dev/null)
FMT_LIBS := $(shell pkg-config --libs fmt 2>/dev/null)

CPPFLAGS += $(FMT_CFLAGS)
CPPFLAGS += -MMD -MP
LDLIBS += $(if $(strip $(FMT_LIBS)),$(FMT_LIBS),-lfmt)

.PHONY: all asan run run-asan gdb clean
.PHONY: format

all: $(TARGET)

DEP = $(OBJ:.o=.d)
-include $(DEP)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD_DIR)/%.o: %.cpp Makefile | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

asan:
	$(MAKE) TARGET=hello_asan BUILD_DIR=build/asan CXXFLAGS+='$(ASAN_CXXFLAGS)' LDFLAGS+='$(ASAN_LDFLAGS)' all

run: $(TARGET)
	./$(TARGET)

run-asan: asan
	./hello_asan

gdb: $(TARGET)
	gdb -q ./$(TARGET)

clean:
	rm -rf build hello hello_asan

format:
	clang-format -i $$(find . -path ./build -prune -o -path ./.git -prune -o \( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o -name '*.h' -o -name '*.hpp' \) -print)
