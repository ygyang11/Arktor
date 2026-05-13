import pytest

from agent_cli.app import _build_parser


def test_mutual_exclusion_continue_and_resume() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-c", "-r", "abc"])


def test_mutual_exclusion_continue_and_session_id() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-c", "-s", "abc"])


def test_mutual_exclusion_resume_and_session_id() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-r", "abc", "-s", "def"])


def test_continue_short_form() -> None:
    assert _build_parser().parse_args(["-c"]).resume_latest is True


def test_continue_long_form() -> None:
    assert _build_parser().parse_args(["--continue"]).resume_latest is True


def test_resume_requires_id() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-r"])


def test_resume_with_id() -> None:
    args = _build_parser().parse_args(["-r", "abc-123"])
    assert args.resume == "abc-123"
    assert args.resume_latest is False
    assert args.session_id is None


def test_session_id_short_alias() -> None:
    assert _build_parser().parse_args(["-s", "abc-123"]).session_id == "abc-123"


def test_session_id_long_alias() -> None:
    assert _build_parser().parse_args(["--session-id", "abc"]).session_id == "abc"


def test_no_flags_defaults() -> None:
    args = _build_parser().parse_args([])
    assert args.resume_latest is False
    assert args.resume is None
    assert args.session_id is None
