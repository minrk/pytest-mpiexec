import os
import sys

import pytest
from mpi4py import MPI


@pytest.mark.parametrize("a", [1, 2])
def test_plain(a):
    assert True


def fail():
    1 / 0


def test_plain_err():
    fail()


@pytest.mark.mpiexec(n=4)
def test_mpi_fail():
    print("mpi fail", MPI.COMM_WORLD.rank, os.getpid())
    assert False


@pytest.mark.mpiexec(n=2)
def test_mpi_error():
    print("mpi error", MPI.COMM_WORLD.rank, os.getpid(), file=sys.stderr)
    fail()


@pytest.mark.mpiexec(n=2)
def test_mpi_ok():
    print("mpi ok", MPI.COMM_WORLD.rank, os.getpid())


@pytest.mark.mpiexec(n=2, timeout=3)
def test_mpi_hang():
    if MPI.COMM_WORLD.rank > 0:
        MPI.COMM_WORLD.Barrier()


@pytest.mark.mpiexec
@pytest.mark.parametrize("mpiexec_n", [1, 2, 3])
@pytest.mark.parametrize("a", [1, 2])
def test_mpi_parametrized(a, mpiexec_n):
    assert MPI.COMM_WORLD.size <= 2, "I don't work with more than 2!"
    assert a == 1
