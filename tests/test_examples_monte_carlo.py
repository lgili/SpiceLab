from pathlib import Path

from examples import monte_carlo_demo


def test_monte_carlo_demo_smoke(tmp_path: Path) -> None:
    outdir = tmp_path / "mc_out"
    outdir.mkdir()
    df = monte_carlo_demo.run_demo(n=4, outdir=outdir, use_real_runner=False)
    # dataframe should be returned
    assert df is not None
    # Expect the histogram file to exist
    files = list(outdir.iterdir())
    assert any(p.name.startswith("monte_carlo_") for p in files)
