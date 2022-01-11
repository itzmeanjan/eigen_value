import similarity_transform as st

N = 1 << 2  # test rounds
DIM = 1 << 10  # test square matrix dimension
TOL = 1e-3  # absoluate error tolerance in floating point math


def main():
    ev = st.EigenValue()  # prepare sycl queue & shared object to be interacted with
    mat = st.np.random.random((DIM, DIM)).astype(
        'f')  # converting dtype to float32
    for i in range(N):
        λ, v, ts, itr = ev.similarity_transform(mat)

        assert st.np.all(st.np.isclose(st.np.matmul(mat, v), λ * v,
                                       atol=TOL)), "Av = λv assertion failed !"
        print(
            f'{i:>3} passed randomized test against {DIM} x {DIM} similarity transform\tin {ts*1e-3:.3f} s\twith {itr} round(s)')


if __name__ == '__main__':
    main()
