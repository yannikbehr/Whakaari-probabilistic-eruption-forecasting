import logging
import os
import subprocess
import sys
import time

import schedule

logger = logging.getLogger(__name__)


def _get_snakefile():
    return os.path.join(os.path.dirname(__file__), "data", "workflows", "Snakefile")


def _run_snakemake(directory, check=False):
    """Invoke snakemake with the bundled workflow in the given directory."""
    snakefile = _get_snakefile()
    subprocess.run(
        [
            "snakemake",
            "--snakefile",
            snakefile,
            "--directory",
            directory,
            "--config",
            f"outdir={directory}",
            "-c1",
        ],
        check=check,
    )


def run_benchmark(directory):
    """Run the snakemake benchmark workflow in the given directory."""
    _run_snakemake(directory, check=True)


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
