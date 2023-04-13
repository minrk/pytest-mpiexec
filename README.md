# pytest-mpixec

pytest plugin for running individual tests with mpiexec

A test marked with `mark.mpiexec` will be run in a separate instance with mpiexec,
instead of in the current process.

The subprocess test will be run with pytest,
so fixtures and everything should still work!

The number of processes can be parametrized if you parametrize an argument called `mpiexec_n`.

## Try it out

```
pip install pytest-mpiexec
```

And write tests that use mpiexec:

```python
from mpi4py import MPI

@pytest.mark.mpiexec(n=2)
def test_my_mpi_code(fixtures):

@pytest.mark.mpiexec
@pytest.mark.parametrize("mpiexec_n", [1, 2, 3])
def test_my_mpi_code(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

```

## Caveats

If you use module or session-scoped fixtures, another instance will be running,
so these can't conflict with other pytest runs (e.g. conflicting on ports, files, etc.)

## Prior art

- pytest-mpi (helpers for tests run inside mpi - compatible with this package!)
- pytest-easyMPI (similar goal to this one, but takes a different approach)
