CXX = clang++
CXXFLAGS = -g --std=c++17 -fsycl
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard include/*.hpp)
INCLUDES = -I./include
PROG = a.out


$(PROG): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) $(HEADERS) $(INCLUDES)
