"""Tests for the LSF batch-script generator."""

import pytest

from transmission_fitter.wexac_utils import generate_lsf_file


@pytest.fixture
def lsf_content(tmp_path):
    """Generate a job file with explicit settings and return its text."""
    path = tmp_path / "example_job.lsf"
    generate_lsf_file(
        job_name="abscal_job",
        queue="short",
        output_file="stdout.%J",
        error_file="stderr.%J",
        command="python calibrate.py --field 923",
        wall_time="02:30",
        num_cores=4,
        memory="8GB",
        lsf_file_name=str(path),
    )
    return path, path.read_text()


class TestGenerateLsfFile:
    def test_writes_the_file_at_the_requested_path(self, lsf_content):
        path, text = lsf_content
        assert path.exists()
        assert text.strip() != ""

    @pytest.mark.parametrize(
        "directive",
        [
            "#BSUB -J abscal_job",
            "#BSUB -q short",
            "#BSUB -W 02:30",
            "#BSUB -o stdout.%J",
            "#BSUB -e stderr.%J",
            'rusage[mem=8GB]',
        ],
    )
    def test_contains_the_expected_bsub_directives(self, lsf_content, directive):
        _, text = lsf_content
        assert directive in text

    def test_contains_the_command_to_run(self, lsf_content):
        _, text = lsf_content
        assert "python calibrate.py --field 923" in text

    def test_defaults_are_applied_when_omitted(self, tmp_path):
        path = tmp_path / "defaults.lsf"
        generate_lsf_file(
            job_name="j",
            queue="q",
            output_file="o",
            error_file="e",
            command="echo hi",
            lsf_file_name=str(path),
        )
        text = path.read_text()
        assert "#BSUB -W 01:00" in text
        assert "rusage[mem=2GB]" in text

    def test_overwrites_an_existing_file(self, tmp_path):
        path = tmp_path / "job.lsf"
        path.write_text("stale content")
        generate_lsf_file(
            job_name="fresh",
            queue="q",
            output_file="o",
            error_file="e",
            command="echo hi",
            lsf_file_name=str(path),
        )
        text = path.read_text()
        assert "stale content" not in text
        assert "#BSUB -J fresh" in text
