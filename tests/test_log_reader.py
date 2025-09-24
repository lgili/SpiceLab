from spicelab.io.log_reader import read_errors


def test_read_errors_finds_percent_error() -> None:
    txt = "INFO ok\n%Error: failure\nwarning only\n%ERROR another one\n"
    errs = read_errors(txt)
    assert len(errs) == 2
