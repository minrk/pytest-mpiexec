import json
import os
import shlex
import subprocess
import sys
from functools import partial
from tempfile import TemporaryDirectory

import pytest

MPI_SUBPROCESS_ENV = "TEST_MPI_SUBTEST"
TEST_REPORT_DIR_ENV = "TEST_MPI_REPORT_DIR"

MPI_MARKER_NAME = "mpiexec"

MPIEXEC = "mpiexec"

pytest_plugins = ["pytest_reportlog"]


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
        reportlog_dir = os.getenv(TEST_REPORT_DIR_ENV, "")

        # TODO: can we guarantee reportlog hasn't already been configured?
        config.option.report_log = os.path.join(
            reportlog_dir, f"reportlog-{rank}.jsonl"
        )


def pytest_collection_modifyitems(session, config, items):
    """Run any tests marked with mpiexec via mpiexec subprocess"""
    if os.getenv(MPI_SUBPROCESS_ENV):
        return
    for item in list(items):
        mpi_mark = item.get_closest_marker(MPI_MARKER_NAME)
        if mpi_mark:
            # run mpi test in a subprocess
            item.runtest = partial(mpi_runtest, item)


def _format_reprentry(reprentry):
    """Format one traceback entry in a pytest longrepr"""
    lines = []
    data = reprentry["data"]
    lines.extend(data["lines"])
    lines.append("")
    lines.append("{path}:{lineno} {message}".format(**data["reprfileloc"]))
    return "\n".join(lines)


def _format_longrepr(longrepr):
    """Format a recorded longrepr

    TODO: figure out how to recover highlighting?
    """
    chunks = []
    for reprentry in longrepr["reprtraceback"]["reprentries"]:
        chunks.append(_format_reprentry(reprentry))
    sep = "\n" + (" -" * 20) + "\n"
    return sep.join(chunks)


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
                f"mpi test did not complete in {timeout} seconds", pytrace=False
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
            if report.get("longrepr"):
                item.add_report_section(
                    report["when"], "traceback", _format_longrepr(report["longrepr"])
                )
            for section in report["sections"]:
                key, rest = section
                key_parts = key.split()
                # import pprint
                # pprint.pprint(report)
                if len(key_parts) == 3 and key_parts[0] == "Captured":
                    key, when = key_parts[1:]
                    key += " (mpi)"
                    # reportlog seems to repeat reports
                    # e.g. captured stdout shows up in teardown and call with the same content
                    if report["when"] == "teardown" and when != report["when"]:
                        continue
                else:
                    key = report["when"]

                item.add_report_section(
                    when,
                    key,
                    rest,
                )
    fail_msg = None
    if p.returncode:
        fail_msg = f"mpi test failed with exit status {p.returncode}"
    elif not reports:
        fail_msg = "No reports captured!"

    if fail_msg:
        if p.stdout:
            item.add_report_section("mpiexec pytest", "stdout", p.stdout)
        if p.stderr:
            item.add_report_section("mpiexec pytest", "stderr", p.stderr)
        pytest.fail(fail_msg, pytrace=False)
