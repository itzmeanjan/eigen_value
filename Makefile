CXX = dpcpp
CXXFLAGS = --std=c++17 -Wall
SYCLFLAGS = -fsycl
AOTFLAGS = -fsycl-default-sub-group-size 32
INCLUDES = -I./include
PROG = run

$(PROG): utils.o similarity_transform.o main.o benchmark_similarity_transform.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

benchmark_similarity_transform.o: benchmarks/benchmark_similarity_transform.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

utils.o: utils.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

similarity_transform.o: similarity_transform.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

main.o: main.cpp
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(INCLUDES) -c $^ -o $@

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
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' -o -name 'lib*.so' | xargs rm -f

aot_cpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c main.cpp -o main.o $(INCLUDES)
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(AOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx512" benchmarks/*.cpp similarity_transform.cpp utils.cpp main.o; \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(AOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx2" benchmarks/*.cpp similarity_transform.cpp utils.cpp main.o; \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(AOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx" benchmarks/*.cpp similarity_transform.cpp utils.cpp main.o; \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(AOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=sse4.2" benchmarks/*.cpp similarity_transform.cpp utils.cpp main.o; \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi

aot_gpu:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(AOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xs "-device 0x4905" similarity_transform.cpp utils.cpp main.o

lib:
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c wrapper/similarity_transform.cpp -fPIC -o wrapper/wrapped_similarity_transform.o
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(INCLUDES) -c similarity_transform.cpp -fPIC -o wrapper/similarity_transform.o
	$(CXX) $(SYCLFLAGS) --shared -fPIC wrapper/similarity_transform.o wrapper/wrapped_similarity_transform.o -o wrapper/libsimilarity_transform.so
