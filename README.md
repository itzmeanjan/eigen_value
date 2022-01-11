# eigen_value
Parallel Eigen value(s)/ vector(s) Computation on Accelerator 

## Background

This is an implementation of maximum eigen value( with respective vector ) computation using **Similarity Transform** method for positive matrices.

Read [paper](https://link.springer.com/chapter/10.1007%2F978-3-319-11194-0_18) for better understanding.

> I keep both sequential & parallel implementation.

## Requirements

- You should have Intel oneAPI basekit installed, which you can find [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).
- Or you may want to build it from source, check [here](https://intel.github.io/llvm-docs/GetStartedGuide.html#prerequisites).
- I'm working on a GNU/ Linux machine, with `make`, `clang-format` installed.

```bash
$ lsb_release -d
Description:    Ubuntu 20.04.3 LTS
```

- My `dpcpp` compiler version is

```bash
$ dpcpp --version
Intel(R) oneAPI DPC++/C++ Compiler 2021.4.0 (2021.4.0.20210924)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/intel/oneapi/compiler/2021.4.0/linux/bin
```

## Benchmark

- Assuming you've all requirements installed on your machine, compile program

```bash
make # or make aot_{cpu,gpu}
```

- Run produced binary

```bash
./run # or ./a.out
```

> Clean up generated object files using `make clean`

> After editing source, you can reformat those using `make format`

### Benchmark Results

I ran parallel implementation of similarity transform algorithm on multiple hardwares, with hilbert matrix of various dimensions, while setting maximum iteration count to *1000* and dynamic work group size.

> Except following ones, I keep detailed benchmark results for all kernels involved in similarity transform method [here](benchmarks/similarity_transform.md).

#### On CPU

```bash
$ make aot_cpu && ./a.out

running on Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

128  x  128			         7 ms			     9 round(s)
256  x  256			         1 ms			    10 round(s)
512  x  512			         3 ms			    12 round(s)
1024 x 1024			         5 ms			    13 round(s)
2048 x 2048			        18 ms			    14 round(s)
4096 x 4096			       101 ms			    15 round(s)
8192 x 8192			       510 ms			    17 round(s)
```

```bash
$ make aot_cpu && ./a.out

running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

128  x  128			        22 ms			     9 round(s)
256  x  256			         6 ms			    10 round(s)
512  x  512			         7 ms			    12 round(s)
1024 x 1024			         8 ms			    13 round(s)
2048 x 2048			        19 ms			    14 round(s)
4096 x 4096			        40 ms			    15 round(s)
8192 x 8192			       126 ms			    17 round(s)
```

```bash
$ make aot_cpu && ./a.out

running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

128  x  128			        11 ms			     9 round(s)
256  x  256			         6 ms			    10 round(s)
512  x  512			        16 ms			    12 round(s)
1024 x 1024			        51 ms			    13 round(s)
2048 x 2048			       200 ms			    14 round(s)
4096 x 4096			       847 ms			    15 round(s)
8192 x 8192			      3759 ms			    17 round(s)
```

```bash
$ make aot_cpu && ./a.out

running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

128  x  128			        18 ms			     9 round(s)
256  x  256			         4 ms			    10 round(s)
512  x  512			         6 ms			    12 round(s)
1024 x 1024			         8 ms			    13 round(s)
2048 x 2048			        27 ms			    14 round(s)
4096 x 4096			        82 ms			    15 round(s)
8192 x 8192			       339 ms			    17 round(s)
```

#### On GPU

```bash
$ make aot_gpu && ./a.out

running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

128  x  128			        10 ms			     9 round(s)
256  x  256			         8 ms			    10 round(s)
512  x  512			        14 ms			    12 round(s)
1024 x 1024			        37 ms			    13 round(s)
2048 x 2048			       138 ms			    14 round(s)
4096 x 4096			       564 ms			    15 round(s)
8192 x 8192			      2509 ms			    17 round(s)
```

```bash
$ make && ./run # JIT compiled kernels, check time column of first row of below table [ way more than second row, because kernel being JIT-ed ]

running on Intel(R) UHD Graphics P630 [0x3e96]

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

128  x  128			       122 ms			     9 round(s)
256  x  256			         9 ms			    10 round(s)
512  x  512			        28 ms			    12 round(s)
1024 x 1024			       104 ms			    13 round(s)
2048 x 2048			       446 ms			    14 round(s)
4096 x 4096			      1858 ms			    15 round(s)
8192 x 8192			      8259 ms			    17 round(s)
```

## Test

A set of minimal test cases are present, which you may want to run using

```bash
make test
```

## Sequential (reference) Implementation

Run sequential implementation using 👇, while ensuring you've `numpy` installed. It also compares computed maximum eigen value with `numpy` computed result.

```bash
python3 -m pip install -U numpy # you may need it
python3 main.py
```

```text
Sequential Similarity Transform, for finding maximum eigen value ( with vector )

32   x   32               1.27 ms                      5 round(s)
64   x   64               2.44 ms                      5 round(s)
128  x  128               9.67 ms                      5 round(s)
256  x  256              20.58 ms                      4 round(s)
512  x  512              74.24 ms                      4 round(s)
1024 x 1024             335.77 ms                      4 round(s)
```

## Python Wrapper

I provide you with one build recipe which can be used for compiling Parallel Similarity Transform's implementation into dynamically linked shared object.

```bash
# AOT compiled for x86_64 CPU; check Makefile
make lib

# now check 
file wrapper/libsimilarity_transform.so
```

Once shared object is ready, you can now interact with C++ implementation of Maximum EigenValue Finder function from Python wrapper function.

```bash
pushd wrapper/python
python3
```

```python
import similarity_transform as st

# must be positive square matrix of `float32`
#
# this is input data matrix
m = st.np.random.random((16, 16)).astype('f')

ev = st.EigenValue()
λ, v, ts, itr = ev.similarity_transform(m)

# λ = maximum eigen value
# v = eigen vector
# ts = execution time ( in ms )
# itr = iterations required before convergence
```

> You may want to take a look at [test case](https://github.com/itzmeanjan/eigen_value/blob/1e7aec0/wrapper/python/test.py#L8) written using Python wrapper.

There's also one script for running tests on randomly generated positive square matrices.

```bash
python3 test.py
popd
```
