CXX = dpcpp
CXXFLAGS = --std=c++17 -Wall
SYCLFLAGS = -fsycl
INCLUDES = -I./include
PROG = run

# $(PROG): utils.o similarity_transform.o main.o
# 	$(CXX) $(SYCLFLAGS) $^ -o $@

# utils.o: utils.cpp
# 	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

# similarity_transform.o: similarity_transform.cpp
# 	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

# main.o: main.cpp
# 	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

test: tests/$(PROG)
	./tests/$(PROG)

tests/$(PROG): tests/test.o tests/similarity_transform.o tests/utils.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

tests/utils.o: utils.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

tests/similarity_transform.o: similarity_transform.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

tests/test.o: tests/test.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

clean:
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' | xargs rm -f
