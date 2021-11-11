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
make
```

- Run produced binary

```bash
./run
```

> Clean up generated object files using `make clean`

> After editing source, you can reformat those using `make format`

### Benchmark Results

I ran parallel implementation of similarity transform algorithm on multiple hardwares, with hilbert matrix of various dimensions, while setting maximum iteration count to *1000* and work group size to *32*.

#### On CPU

```bash
$ make aot_cpu && ./a.out

running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

32   x   32                             10 ms                        7 round(s)
64   x   64                              3 ms                        8 round(s)
128  x  128                              3 ms                        9 round(s)
256  x  256                              5 ms                       10 round(s)
512  x  512                             15 ms                       12 round(s)
1024 x 1024                             49 ms                       13 round(s)
2048 x 2048                            186 ms                       14 round(s)
4096 x 4096                            775 ms                       15 round(s)
8192 x 8192                           3461 ms                       17 round(s)
```

```bash
$ make && ./run # JIT compiled kernels

running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

32   x   32                            341 ms                        7 round(s)
64   x   64                              2 ms                        8 round(s)
128  x  128                              3 ms                        9 round(s)
256  x  256                              4 ms                       10 round(s)
512  x  512                              6 ms                       12 round(s)
1024 x 1024                             12 ms                       13 round(s)
2048 x 2048                             34 ms                       14 round(s)
4096 x 4096                            106 ms                       15 round(s)
8192 x 8192                            422 ms                       17 round(s)
```

#### On GPU

```bash
$ make aot_gpu && ./a.out

running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

32   x   32                            120 ms                        7 round(s)
64   x   64                              4 ms                        8 round(s)
128  x  128                              4 ms                        9 round(s)
256  x  256                              6 ms                       10 round(s)
512  x  512                             13 ms                       12 round(s)
1024 x 1024                             57 ms                       13 round(s)
2048 x 2048                            247 ms                       14 round(s)
4096 x 4096                           1251 ms                       15 round(s)
8192 x 8192                           7358 ms                       17 round(s)
```

```bash
$ make && ./run # JIT compiled kernels

running on Intel(R) UHD Graphics P630 [0x3e96]

Parallel Algorithm using Similarity Transform for finding max eigen value (with vector)

32   x   32                            129 ms                        7 round(s)
64   x   64                              5 ms                        8 round(s)
128  x  128                              5 ms                        9 round(s)
256  x  256                              6 ms                       10 round(s)
512  x  512                             10 ms                       12 round(s)
1024 x 1024                             29 ms                       13 round(s)
2048 x 2048                            155 ms                       14 round(s)
4096 x 4096                            931 ms                       15 round(s)
8192 x 8192                           5995 ms                       17 round(s)
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
