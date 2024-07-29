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
  assert something

@pytest.mark.mpiexec
@pytest.mark.parametrize("mpiexec_n", [1, 2, 3])
def test_my_mpi_code(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n

```

## Options

If your mpi executable is not `mpiexec` for some reason, you can specify it with:

```
pytest --mpiexec your-mpiexec
```

pytest-mpiexec tries to reduce noise while still showing useful info for failures.
That means only one output should appear when everything's working.
But parallel failures can be both noisy and confusing, so there are options here,
exposed via the `--mpiexec-report` option:

| option          | behavior                                                            |
| --------------- | ------------------------------------------------------------------- |
| `first_failure` | show the test output of the first rank with an error (**default**)  |
| `all_failures`  | show output from _all_ ranks that fail (often noisy)                |
| `all`           | show _all_ test results (even duplicate PASSED outputs)             |
| `concise`       | try to show only each _unique_ failure only once (**experimental**) |

The `concise` option is experimental and aims to strike a balance between the possibly omitted information from `first_failure` and the redundant noise of `all_failures`.

## Caveats

If you use module or session-scoped fixtures, another instance will be running,
so these can't conflict with other pytest runs (e.g. conflicting on ports, files, etc.)

## Prior art

- pytest-mpi (helpers for tests run inside mpi - compatible with this package!)
- pytest-easyMPI (similar goal to this one, but takes a different approach)
