import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import schedule

from whakaaribn import get_data

logger = logging.getLogger(__name__)


BACKEND_SNAKEFILES = {
    "smile": get_data("data/workflow/smile_workflow.smk"),
    "pgmpy": get_data("data/workflow/pgmpy_workflow.smk"),
}


def _prepare_workflow_directory(snakefile: str, directory: str) -> str:
    """Copy the snakefile, rules/, and notebooks/ into *directory* if it differs
    from the bundled workflow directory.  Returns the path to the snakefile that
    should be passed to snakemake (the copy when applicable, original otherwise).
    """
    workflow_dir = Path(get_data("data/workflow"))
    target_dir = Path(directory)

    if target_dir.resolve() == workflow_dir.resolve():
        return snakefile  # Nothing to do

    target_dir.mkdir(parents=True, exist_ok=True)

    for subdir in ("rules", "notebooks"):
        src = workflow_dir / subdir
        dst = target_dir / subdir
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    dst_snakefile = target_dir / Path(snakefile).name
    shutil.copy2(snakefile, dst_snakefile)
    return str(dst_snakefile)


def _run_snakemake(snakefile: str = None, directory: str = None, cores: int = 1,
                   check=False, extra_args: Sequence[str] | None = None,
                   backend: str = "smile", clean: bool = False):
    """Invoke snakemake with the bundled workflow in the given directory."""
    if snakefile is None:
        snakefile = BACKEND_SNAKEFILES.get(backend)
        if snakefile is None:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: {list(BACKEND_SNAKEFILES)}.")
    if directory is None:
        directory = get_data("data/workflow")
    snakefile = _prepare_workflow_directory(snakefile, directory)
    cmd = [
        "snakemake",
        "--snakefile",
        snakefile,
        "--directory",
        directory,
        "--cores",
        str(cores),
    ]
    if clean:
        cmd.append("--delete-all-output")
    if extra_args:
        cmd.extend(extra_args)
    try:
        subprocess.run(
            cmd,
            check=check,
        )
    except Exception as e:
        print(e)
        sys.exit(1)
    sys.exit(0)


def run_benchmark(directory, backend="smile", cores=1, clean=False):
    """Run the snakemake benchmark workflow in the given directory."""
    _run_snakemake(directory=directory, check=True,
                   backend=backend, cores=cores, clean=clean)


def _run_workflow(directory, backend="smile", cores=1, clean=False):
    """Run the snakemake workflow in the given directory."""
    try:
        _run_snakemake(directory=directory, check=True,
                       backend=backend, cores=cores, clean=clean)
    except subprocess.CalledProcessError as e:
        logger.error("Workflow run failed: %s", e)


def daemon(directory, backend="smile", cores=1, clean=False):
    """Start a scheduled job running the workflow in regular intervals."""
    _run_workflow(directory, backend=backend, cores=cores, clean=clean)
    schedule.every().day.at("13:00").do(
        _run_workflow, directory=directory, backend=backend, cores=cores, clean=clean)

    while True:
        schedule.run_pending()
        time.sleep(60)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Whakaari BN command line interface")
    subparsers = parser.add_subparsers(dest="command")

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run the benchmark workflow"
    )
    benchmark_parser.add_argument(
        "--directory",
        required=True,
        help="Directory to run the benchmark in",
    )
    benchmark_parser.add_argument(
        "--backend",
        choices=list(BACKEND_SNAKEFILES),
        default="pgmpy",
        help="Backend to use for the workflow [default: pgmpy]",
    )
    benchmark_parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores to pass to snakemake [default: 1]",
    )
    benchmark_parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Delete all output files before running (passes --delete-all-output to snakemake)",
    )

    daemon_parser = subparsers.add_parser(
        "daemon", help="Start a scheduled job running the workflow in regular intervals"
    )
    daemon_parser.add_argument(
        "--directory",
        default="/opt/data",
        help="Directory to run the workflow in [default: /opt/data]",
    )
    daemon_parser.add_argument(
        "--backend",
        choices=list(BACKEND_SNAKEFILES),
        default="pgmpy",
        help="Backend to use for the workflow [default: pgmpy]",
    )
    daemon_parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="Number of cores to pass to snakemake [default: 1]",
    )
    daemon_parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Delete all output files before running (passes --delete-all-output to snakemake)",
    )

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        run_benchmark(args.directory, backend=args.backend,
                      cores=args.cores, clean=args.clean)
    elif args.command == "daemon":
        daemon(args.directory, backend=args.backend,
               cores=args.cores, clean=args.clean)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
