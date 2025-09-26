import pytest


def _make_pandas_df():
    try:
        import pandas as pd
    except Exception:
        pytest.skip("pandas not installed; skipping pandas-backed test")
    df = pd.DataFrame({"V(vout)": [1.23, 4.56], "other": [7, 8]})
    return df


class FakeMC:
    def __init__(self, samples, df):
        self.samples = samples
        self._df = df

    def to_dataframe(self, metric=None):
        return self._df


def test_orchestrator_mc_with_pandas_and_metric_col(monkeypatch):
    df = _make_pandas_df()
    samples = [{"R": 1000.0}, {"R": 1100.0}]
    fake_mc = FakeMC(samples, df)

    def fake_monte_carlo(
        circuit,
        mapping,
        n,
        *,
        analyses,
        engine=None,
        workers=None,
        cache_dir=None,
    ):
        return fake_mc

    import examples.monte_carlo_demo as demo_mod

    monkeypatch.setattr(demo_mod, "monte_carlo", fake_monte_carlo)

    # call _orchestrator_mc with metric_col explicitly set to 'V(vout)'
    from examples.monte_carlo_demo import _orchestrator_mc

    samples_out, metrics = _orchestrator_mc(2, "ngspice", metric_col="V(vout)")
    assert len(samples_out) == len(samples)
    for out, expected in zip(samples_out, samples, strict=False):
        assert out["R"] == expected["R"]
        assert "C" in out
    assert len(metrics) == 2
    assert abs(metrics[0] - 1.23) < 1e-9
