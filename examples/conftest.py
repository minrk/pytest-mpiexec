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


def pytest_configure(config):
    config.addinivalue_line("markers", "mpi: Run this text with mpiexec")
    if not os.getenv(MPI_SUBPROCESS_ENV):
        return
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.rank
    reportlog_dir = os.getenv(TEST_REPORT_DIR_ENV, "")

    # TODO: can we guarantee reportlog hasn't already been configured?
    config.option.report_log = os.path.join(reportlog_dir, f"reportlog-{rank}.jsonl")


def pytest_pycollect_makeitem(collector, name, obj):
    if os.getenv(MPI_SUBPROCESS_ENV):
        return
    return
    pytestmark = getattr(obj, "pytestmark", None)
    if not pytestmark:
        return
    mpi_mark = None
    for mark in pytestmark[::-1]:
        if mark.name == "mpi":
            mpi_mark = mark
            break
    if mpi_mark is None:
        return None

    n_list = mpi_mark.kwargs.get("n", 2)
    if isinstance(n_list, int):
        n_list = [n_list]
    items = []
    for n in n_list:
        f = pytest.Function.from_parent(
            collector,
            name=f"{name}[n={n}]",
            n=n,
            callobj=obj,
            real_nodeid=f"{collector.fspath}::{name}",
        )
        # f.runtest = partial(mpi_runtest, n, f, nodeid=f"{collector.fspath}::{name}")
        print("adding ", f.nodeid)
        items.append(f)
    return items


# def pytest_generate_tests(metafunc):
#     print(f"{metafunc=}, {metafunc.function}")
#     obj = metafunc.function
#     pytestmark = getattr(obj, "pytestmark", None)
#     if not pytestmark:
#         return
#     mpi_mark = None
#     for mark in pytestmark[::-1]:
#         if mark.name == "mpi":
#             mpi_mark = mark
#             break
#     if mpi_mark is None:
#         return None
#     n_list = mpi_mark.kwargs.get("n", 2)
#     if isinstance(n_list, int):
#         n_list = [n_list]
#     # metafunc.function = partial(mpi_runtest
#     items = metafunc.parametrize(argnames=["mpi_n"], argvalues=[n_list], indirect=True)
#     return items


class MPIFunction(pytest.Function):
    _ALLOW_MARKERS = True

    def __init__(self, n, real_nodeid, callobj, **kwargs):
        super().__init__(**kwargs)
        self._n = n
        self._real_nodeid = real_nodeid

    def runtest(self):
        mpi_runtest(self._n, nodeid=self._real_nodeid, item=self)


#     if n and i
#     mpi_mark
#     if inspect.isfunction(obj) and \
# 62            name.startswith("test_"):
# 64        f = pytest.Function(name, parent=collector)
# 65    else:
# 66        return []


def pytest_collection_modifyitems(session, config, items):
    if os.getenv(MPI_SUBPROCESS_ENV):
        return
    for item in list(items):
        mpi_mark = item.get_closest_marker("mpi")
        # print(item.nodeid)
        if mpi_mark:
            # run mpi test in a subprocess
            item.runtest = partial(mpi_runtest, item)

            n = mpi_mark.kwargs.get("n", 2)
            common_kwargs = {}
            common_kwargs.update(mpi_mark.kwargs)
            common_kwargs.pop("n", None)


#
#             if isinstance(n, (list, tuple)):
#                 n_list = n
#                 idx = items.index(item)
#                 items.remove(item)
#                 for n in n_list:
#                     item_n = copy.copy(item)
#                     item_n.runtest = partial(mpi_runtest, item_n)
#                     item_n.nodeid += f"[mpi_n={n}]"
#                     item_n.add_marker(pytest.mark.mpi(n=n, **common_kwargs))
#                     # register new item
#                     print("adding", item_n)
#                     items.insert(idx, item_n)


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
    mpi_mark = item.get_closest_marker("mpi")
    # allow parametrization
    if "mpi_n" in item.callspec.params:
        n = item.callspec.params["mpi_n"]
    else:
        n = mpi_mark.kwargs.get("n", 2)
    timeout = mpi_mark.kwargs.get("timeout", 30)

    reportlog = "reportlog.json"
    exe = [
        "mpiexec",
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
        "setup", "mpi exec command", f"{MPI_SUBPROCESS_ENV}=1 {shlex.join(exe)}"
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
