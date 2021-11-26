## Benchmarking Similarity Transform Kernels

Following are benchmark results I get after running all kernel benchmarks on CPU, GPU.

### On CPU

```bash
make aot_cpu && ./a.out
```

```bash
running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Parallel Similarity Transform for finding max eigen value (with vector)

128  x  128			        12 ms			     9 round(s)
256  x  256			         6 ms			    10 round(s)
512  x  512			        16 ms			    12 round(s)
1024 x 1024			        51 ms			    13 round(s)
2048 x 2048			       197 ms			    14 round(s)
4096 x 4096			       846 ms			    15 round(s)
8192 x 8192			      3741 ms			    17 round(s)

Parallel Sum Across Rows of Matrix

128  x  128			      0.29 ms
256  x  256			     0.311 ms
512  x  512			     0.799 ms
1024 x 1024			      2.06 ms
2048 x 2048			     8.082 ms
4096 x 4096			    33.077 ms
8192 x 8192			   127.637 ms

Parallel Max Value in Vector

  128			     0.189 ms
  256			     0.125 ms
  512			     0.156 ms
 1024			     0.127 ms
 2048			     0.135 ms
 4096			     0.141 ms
 8192			     0.146 ms

Parallel Eigen Vector Computation

  128			     0.067 ms
  256			     0.057 ms
  512			      0.24 ms
 1024			     0.061 ms
 2048			     0.044 ms
 4096			     0.061 ms
 8192			     0.058 ms

Parallel Next Matrix Computation

  128			     0.105 ms
  256			     0.198 ms
  512			     0.444 ms
 1024			     1.388 ms
 2048			     5.396 ms
 4096			    21.388 ms
 8192			    85.002 ms

Parallel Stop Criteria Checker

  128			     0.205 ms
  256			     0.125 ms
  512			     0.131 ms
 1024			     0.144 ms
 2048			     0.297 ms
 4096			     0.208 ms
 8192			     0.194 ms
```

### On GPU

```bash
make aot_gpu && ./a.out
```

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

Parallel Similarity Transform for finding max eigen value (with vector)

128  x  128			        14 ms			     9 round(s)
256  x  256			         8 ms			    10 round(s)
512  x  512			        15 ms			    12 round(s)
1024 x 1024			        36 ms			    13 round(s)
2048 x 2048			       131 ms			    14 round(s)
4096 x 4096			       534 ms			    15 round(s)
8192 x 8192			      2372 ms			    17 round(s)

Parallel Sum Across Rows of Matrix

128  x  128			     0.567 ms
256  x  256			      0.65 ms
512  x  512			       1.3 ms
1024 x 1024			     3.949 ms
2048 x 2048			    14.079 ms
4096 x 4096			    54.403 ms
8192 x 8192			   216.512 ms

Parallel Max Value in Vector

  128			     0.525 ms
  256			     0.304 ms
  512			     0.294 ms
 1024			       0.3 ms
 2048			       0.3 ms
 4096			     0.299 ms
 8192			     0.291 ms

Parallel Eigen Vector Computation

  128			     0.054 ms
  256			     0.063 ms
  512			     0.049 ms
 1024			     0.122 ms
 2048			     0.053 ms
 4096			     0.052 ms
 8192			     0.061 ms

Parallel Next Matrix Computation

  128			     0.055 ms
  256			      0.06 ms
  512			     0.086 ms
 1024			     0.193 ms
 2048			     0.577 ms
 4096			     2.414 ms
 8192			      9.94 ms

Parallel Stop Criteria Checker

  128			     0.558 ms
  256			     0.344 ms
  512			     0.308 ms
 1024			     0.302 ms
 2048			     0.306 ms
 4096			     0.313 ms
 8192			     0.307 ms
```
