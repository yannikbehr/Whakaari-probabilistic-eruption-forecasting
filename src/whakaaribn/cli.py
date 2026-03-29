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

    for subdir in ("notebooks",):
        src = workflow_dir / subdir
        dst = target_dir / subdir
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    dst_snakefile = target_dir / Path(snakefile).name
    shutil.copy2(snakefile, dst_snakefile)
    return str(dst_snakefile)


def _run_snakemake(
    snakefile: str = None,
    directory: str = None,
    cores: int = 1,
    check=False,
    extra_args: Sequence[str] | None = None,
    clean: bool = False,
):
    """Invoke snakemake with the bundled workflow in the given directory."""
    if snakefile is None:
        snakefile = get_data("data/workflow/Snakefile")
        if snakefile is None:
            raise ValueError("Can't find pipeline file")
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
    subprocess.run(
        cmd,
        check=check,
    )


def _unlock_snakemake(snakefile: str, directory: str) -> None:
    """Run ``snakemake --unlock`` to release a stale lock from an interrupted run."""
    cmd = ["snakemake", "--snakefile", snakefile, "--directory", directory, "--unlock"]
    try:
        subprocess.run(cmd, check=False)
    except Exception as e:
        logger.warning("snakemake --unlock failed: %s", e)


def run_benchmark(directory, backend="pgmpy", cores=1, clean=False):
    """Run the snakemake benchmark workflow in the given directory."""
    _run_snakemake(
        directory=directory,
        check=True,
        cores=cores,
        clean=clean,
        extra_args=["--config", f"model={backend}", f"n_jobs={cores}"],
    )


def _run_monitoring_workflow(directory, cores=1, clean=False):
    """Run the snakemake workflow in the given directory."""
    try:
        snakefile = get_data("data/workflow/monitoring_pipeline.smk")
        resolved_snakefile = _prepare_workflow_directory(snakefile, directory)
        _unlock_snakemake(resolved_snakefile, directory)
        _run_snakemake(
            snakefile=snakefile,
            directory=directory,
            check=True,
            cores=cores,
            clean=clean,
            extra_args=["--forcerun", "live_data"],
        )
    except subprocess.CalledProcessError as e:
        logger.error("Workflow run failed: %s", e)


def daemon(directory, backend="pgmpy", cores=1, clean=False):
    """Start a scheduled job running the workflow in regular intervals."""
    _run_monitoring_workflow(directory, cores=cores, clean=clean)
    schedule.every().day.at("13:00").do(
        _run_monitoring_workflow, directory=directory, cores=cores, clean=clean
    )

    while True:
        schedule.run_pending()
        time.sleep(60)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Whakaari BN command line interface")
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
        choices=("pgmpy", "smile"),
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
        run_benchmark(
            args.directory, backend=args.backend, cores=args.cores, clean=args.clean
        )
    elif args.command == "daemon":
        daemon(args.directory, backend=args.backend, cores=args.cores, clean=args.clean)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
