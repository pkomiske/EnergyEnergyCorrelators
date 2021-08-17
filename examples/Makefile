CXX = g++
SRCS = basic_example.cpp
CXXFLAGS = -O3 -Wall -std=c++14 -g -ffast-math

FASTJETINC = $(shell fastjet-config --cxxflags)
FASTJETLIB = $(shell fastjet-config --libs)
INCLUDES  += $(FASTJETINC) -I../eec/include -I../EventGeometry -I../EventGeometry/Wasserstein
LIBRARIES += $(FASTJETLIB) -lz

CNPY = ../Wasserstein/examples/src/cnpy.o

# http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#combine
DEPDIR = .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(INCLUDES) -c
%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

.PHONY: all clean
all: $(SRCS:.cpp=.o) basic_example

basic_example: basic_example.o $(CNPY)
	$(CXX) -o $@ $^ $(LIBRARIES)

$(CNPY):
	make -C ../EventGeometry/Wasserstein/examples src/cnpy.o

clean:
	rm -rfv *.o *_example $(DEPDIR)

$(DEPDIR): ; @mkdir -p $@
DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))
