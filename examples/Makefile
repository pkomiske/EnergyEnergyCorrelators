SRCS = cnpy.cpp basic_example.cpp
CXXFLAGS = -O3 -Wall -std=c++14 -g

FASTJETINC = $(shell fastjet-config --cxxflags)
FASTJETLIB = $(shell fastjet-config --libs)
INCLUDES  += $(FASTJETINC) -I../eec/include
LIBRARIES += $(FASTJETLIB) -lz

# http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#combine
DEPDIR = .deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CXXFLAGS) $(INCLUDES) -c
%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cpp) $(OUTPUT_OPTION) $<

.PHONY: all clean
all: $(SRCS:.cpp=.o) basic_example

basic_example: basic_example.o cnpy.o
	$(CXX) -o $@ $^ $(LIBRARIES)

clean:
	rm -fv *.o

$(DEPDIR): ; @mkdir -p $@
DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d)
$(DEPFILES):

include $(wildcard $(DEPFILES))
