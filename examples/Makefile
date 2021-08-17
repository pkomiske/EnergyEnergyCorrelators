CXX = g++
SRCS = $(shell ls *.cpp) cnpy.cpp
CXXFLAGS = -O3 -Wall -std=c++14 -g -ffast-math

FASTJETINC = $(shell fastjet-config --cxxflags)
FASTJETLIB = $(shell fastjet-config --libs)
INCLUDES  += $(FASTJETINC) -I../eec/include -Iinclude
LIBRARIES += $(FASTJETLIB) -lz

ifeq ($(shell uname), Darwin)
	CXXFLAGS += -Xpreprocessor -fopenmp
	LIBRARIES += -lomp
endif
ifeq ($(shell uname), Linux)
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

# http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#combine
DEPDIR = .deps
COMPILE.cpp = $(CXX) $(CXXFLAGS) $(INCLUDES) -c $(<) -o $(@) -MT $(@) -MMD -MP -MF $(DEPDIR)/$*.d
%.o : %.cpp
src/%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp)
src/%.o : src/%.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp)

.PHONY: all clean
all: basic_example

basic_example: src/basic_example.o src/cnpy.o
	$(CXX) -o $@ $^ $(LIBRARIES)

clean:
	rm -rfv *.o src/*.o *_example $(DEPDIR)

$(DEPDIR): ; @mkdir -p $@
DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))
