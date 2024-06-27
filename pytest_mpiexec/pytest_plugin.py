import json
import os
import shlex
import subprocess
import sys
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pytest_reportlog.plugin import ReportLogPlugin

MPI_SUBPROCESS_ENV = "TEST_MPI_SUBTEST"
TEST_REPORT_DIR_ENV = "TEST_MPI_REPORT_DIR"

MPI_MARKER_NAME = "mpiexec"

MPIEXEC = "mpiexec"


def pytest_addoption(parser):
    parser.addoption(
        "--mpiexec",
        action="store",
        default=None,
        help="Name of program to run MPI, e.g. mpiexec",
    )


def pytest_configure(config):
    global MPIEXEC
    mpiexec = config.getoption("--mpiexec")
    if mpiexec:
        MPIEXEC = mpiexec

    config.addinivalue_line("markers", f"{MPI_MARKER_NAME}: Run this text with mpiexec")
    if os.getenv(MPI_SUBPROCESS_ENV):
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.rank
        reportlog_dir = Path(os.getenv(TEST_REPORT_DIR_ENV, ""))
        report_path = reportlog_dir / f"reportlog-{rank}.jsonl"
        config._mpiexec_reporter = reporter = ReportLogPlugin(config, report_path)
        config.pluginmanager.register(reporter)


def pytest_unconfigure(config):
    reporter = getattr(config, "_mpiexec_reporter", None)
    if reporter:
        reporter.close()


def mpi_runtest_protocol(item):
    """The runtest protocol for mpi tests

    Runs the test in an mpiexec subprocess

    instead of the current process
    """
    config = item.config
    hook = item.config.hook
    hook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    call = pytest.CallInfo.from_call(partial(mpi_runtest, item), "setup")
    if call.excinfo:
        report = hook.pytest_runtest_makereport(item=item, call=call)
        hook.pytest_runtest_logreport(report=report)
    hook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)


def pytest_runtest_protocol(item, nextitem):
    """Run the MPI protocol for mpi tests

    otherwise, do nothing
    """
    if os.getenv(MPI_SUBPROCESS_ENV):
        return
    mpi_mark = item.get_closest_marker(MPI_MARKER_NAME)
    if not mpi_mark:
        return
    mpi_runtest_protocol(item)
    return True


def mpi_runtest(item):
    """Replacement for runtest

    Runs a single test with mpiexec
    """
    mpi_mark = item.get_closest_marker(MPI_MARKER_NAME)
    # allow parametrization
    if getattr(item, "callspec", None) and "mpiexec_n" in item.callspec.params:
        n = item.callspec.params["mpiexec_n"]
    else:
        n = mpi_mark.kwargs.get("n", 2)
    timeout = mpi_mark.kwargs.get("timeout", 30)
    exe = [
        MPIEXEC,
        "-n",
        str(n),
        sys.executable,
        "-m",
        "pytest",
        "--quiet",
        "--no-header",
        "--no-summary",
        f"{item.fspath}::{item.name}",
    ]
    env = dict(os.environ)
    env[MPI_SUBPROCESS_ENV] = "1"
    # add the mpiexec command for easy re-run
    item.add_report_section(
        "setup", "mpiexec command", f"{MPI_SUBPROCESS_ENV}=1 {shlex.join(exe)}"
    )

    with TemporaryDirectory() as reportlog_dir:
        env[TEST_REPORT_DIR_ENV] = reportlog_dir
        try:
            p = subprocess.run(
                exe,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            if e.stdout:
                item.add_report_section(
                    "mpiexec pytest", "stdout", e.stdout.decode("utf8", "replace")
                )
            if e.stderr:
                item.add_report_section(
                    "mpiexec pytest", "stderr", e.stderr.decode("utf8", "replace")
                )
            pytest.fail(
                f"mpi test did not complete in {timeout} seconds",
                pytrace=False,
            )

        reportlog_root = os.path.join(reportlog_dir, "reportlog-0.jsonl")
        reports = []
        if os.path.exists(reportlog_root):
            with open(reportlog_root) as f:
                for line in f:
                    reports.append(json.loads(line))

    # collect report items for the test
    for report in reports:
        if report["$report_type"] == "TestReport":
            # reconstruct and redisplay the report
            r = item.config.hook.pytest_report_from_serializable(
                config=item.config, data=report
            )
            item.config.hook.pytest_runtest_logreport(config=item.config, report=r)

    if p.returncode or not reports:
        if p.stdout:
            item.add_report_section("mpiexec pytest", "stdout", p.stdout)
        if p.stderr:
            item.add_report_section("mpiexec pytest", "stderr", p.stderr)
    if not reports:
        pytest.fail("No test reports captured from mpi subprocess!", pytrace=False)
