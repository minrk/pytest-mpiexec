import pytest
from mpi4py import MPI


@pytest.mark.mpiexec
@pytest.mark.parametrize("mpiexec_n", [1, 2, 3])
def test_my_mpi_code(mpiexec_n):
    assert MPI.COMM_WORLD.size == mpiexec_n
