CXX = dpcpp
CXXFLAGS = --std=c++17 -Wall
SYCLFLAGS = -fsycl
INCLUDES = -I./include
PROG = run

$(PROG): utils.o similarity_transform.o main.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

utils.o: utils.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

similarity_transform.o: similarity_transform.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

main.o: main.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

clean:
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' | xargs rm -f
