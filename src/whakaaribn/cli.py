import logging
import subprocess
import sys
import time
from typing import Sequence

import schedule
from snakemake.cli import args_to_api, parse_args

from whakaaribn import get_data

logger = logging.getLogger(__name__)


def _run_snakemake(snakefile: str = None, directory: str = None, cores: int = 1, check=False, extra_args: Sequence[str] | None = None):
    """Invoke snakemake with the bundled workflow in the given directory."""
    if snakefile is None:
        snakefile = get_data("data/workflows/Snakefile")
    if directory is None:
        directory = get_data("data/workflows")
    cmd = [
        "--snakefile",
        snakefile,
        "--directory",
        directory,
        f"-c{cores}",
    ]
    if extra_args:
        cmd.extend(extra_args)
    try:
        parser, args = parse_args(cmd)
        success = args_to_api(args, parser)
    except Exception as e:
        print(e)
        sys.exit(1)
    sys.exit(0 if success else 1)
    # subprocess.run(
    #     cmd,
    #     check=check,
    # )


def run_benchmark(directory):
    """Run the snakemake benchmark workflow in the given directory."""
    snakefile = get_data("data/workflows/Snakefile")
    _run_snakemake(snakefile=snakefile, directory=directory, check=True)


def _run_workflow(directory):
    """Run the snakemake workflow in the given directory."""
    try:
        _run_snakemake(directory, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Workflow run failed: %s", e)


def daemon(directory):
    """Start a scheduled job running the workflow in regular intervals."""
    _run_workflow(directory)
    schedule.every().day.at("13:00").do(_run_workflow, directory=directory)

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

    daemon_parser = subparsers.add_parser(
        "daemon", help="Start a scheduled job running the workflow in regular intervals"
    )
    daemon_parser.add_argument(
        "--directory",
        default="/opt/data",
        help="Directory to run the workflow in [default: /opt/data]",
    )

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        run_benchmark(args.directory)
    elif args.command == "daemon":
        daemon(args.directory)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
