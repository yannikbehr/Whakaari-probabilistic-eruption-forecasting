import os
import subprocess
from unittest.mock import patch

import pytest

from whakaaribn.cli import _get_snakefile, _run_snakemake, _run_workflow, daemon, main, run_benchmark


def test_get_snakefile_exists():
    snakefile = _get_snakefile()
    assert os.path.isfile(snakefile), f"Snakefile not found at {snakefile}"


def test_main_no_args_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 1


def test_main_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "benchmark" in captured.out
    assert "daemon" in captured.out


def test_benchmark_missing_directory():
    with pytest.raises(SystemExit):
        main(["benchmark"])


def test_run_snakemake_calls_snakemake(tmp_path):
    with patch("whakaaribn.cli.subprocess.run") as mock_run:
        _run_snakemake(str(tmp_path))
        assert mock_run.called
        args = mock_run.call_args[0][0]
        assert args[0] == "snakemake"
        assert "--directory" in args
        assert str(tmp_path) in args


def test_run_snakemake_check_flag(tmp_path):
    with patch("whakaaribn.cli.subprocess.run") as mock_run:
        _run_snakemake(str(tmp_path), check=True)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs.get("check") is True


def test_run_benchmark_uses_check(tmp_path):
    with patch("whakaaribn.cli._run_snakemake") as mock_snakemake:
        run_benchmark(str(tmp_path))
        mock_snakemake.assert_called_once_with(str(tmp_path), check=True)


def test_run_workflow_logs_error_on_failure(tmp_path):
    with patch("whakaaribn.cli._run_snakemake", side_effect=subprocess.CalledProcessError(1, "snakemake")):
        with patch("whakaaribn.cli.logger") as mock_logger:
            _run_workflow(str(tmp_path))
            mock_logger.error.assert_called_once()


def test_main_benchmark_subcommand(tmp_path):
    with patch("whakaaribn.cli.run_benchmark") as mock_run:
        main(["benchmark", "--directory", str(tmp_path)])
        mock_run.assert_called_once_with(str(tmp_path))


def test_main_daemon_subcommand():
    with patch("whakaaribn.cli.daemon") as mock_daemon:
        main(["daemon", "--directory", "/some/dir"])
        mock_daemon.assert_called_once_with("/some/dir")


def test_main_daemon_default_directory():
    with patch("whakaaribn.cli.daemon") as mock_daemon:
        main(["daemon"])
        mock_daemon.assert_called_once_with("/opt/data")
