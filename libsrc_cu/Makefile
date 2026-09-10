# Static library for CUDA/C++ code

# Compiler
NVCC    := nvcc

# Source/include directories
SRCDIR  := src/
INCDIR  := include/

# Compilation flags
# CFLAGS  := -O3 -std=c++14 -I $(INCDIR) -c -arch=sm_86 -rdc=true
CFLAGS  := -g -std=c++14 -I $(INCDIR) -c -arch=sm_86 -rdc=true

# Library name + version
LIB_NAME := bcpnn
LIB_VER  := 2.2
LIB_FILE := lib$(LIB_NAME)-$(LIB_VER).a

# Object directory
OBJDIR := bin

# Source files (no main/TNetw/LTSL)
SRCS_CPP := \
	$(SRCDIR)Parseparam.cpp \
	$(SRCDIR)Pop.cpp \
	$(SRCDIR)TDPop.cpp \
	$(SRCDIR)PatternFactory.cpp \
	$(SRCDIR)Logger.cpp \
	$(SRCDIR)AxoDelay.cpp \
	$(SRCDIR)Globals.cpp \
	$(SRCDIR)Netw.cpp \
	$(SRCDIR)Prjbase.cpp \
	$(SRCDIR)Prj.cpp \
	$(SRCDIR)Prj1.cpp \
	$(SRCDIR)Prj2.cpp \
	$(SRCDIR)Prj3.cpp \
	$(SRCDIR)Analys.cpp \
	$(SRCDIR)LTSL.cpp \
	$(SRCDIR)LTSLNetw.cpp \
	$(SRCDIR)Probe.cpp

SRCS_CU := \
	$(SRCDIR)GPUGlobals.cu \
	$(SRCDIR)Pop_cu.cu \
	$(SRCDIR)TDPop_cu.cu \
	$(SRCDIR)Prj_cu.cu \
	$(SRCDIR)Prj1_cu.cu
#	$(SRCDIR)Prj2_cu.cu

SRCS := $(SRCS_CPP) $(SRCS_CU)

# Object mapping
OBJS_CPP := $(patsubst $(SRCDIR)%.cpp,$(OBJDIR)/%.o,$(SRCS_CPP))
OBJS_CU  := $(patsubst $(SRCDIR)%.cu,$(OBJDIR)/%.o,$(SRCS_CU))
OBJS     := $(OBJS_CPP) $(OBJS_CU)

# Default: build library
all: $(LIB_FILE)

# Ensure bin/ exists
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile .cpp
$(OBJDIR)/%.o: $(SRCDIR)%.cpp | $(OBJDIR)
	$(NVCC) $(CFLAGS) $< -o $@

# Compile .cu
$(OBJDIR)/%.o: $(SRCDIR)%.cu | $(OBJDIR)
	$(NVCC) $(CFLAGS) $< -o $@

# Create static library
$(LIB_FILE): $(OBJS)
	ar rcs $@ $^

# Clean
clean:
	rm -rf $(OBJDIR) $(LIB_FILE)

.PHONY: all clean
