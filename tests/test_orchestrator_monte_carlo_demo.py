import types


class FakeMC:
    def __init__(self, samples, df):
        self.samples = samples
        self._df = df

    def to_dataframe(self, metric=None):
        return self._df


def test_orchestrator_mc_column_choice(monkeypatch):
    # build a fake dataframe-like object (pandas-like) with numeric 'V(vout)'
    class DF:
        def __init__(self):
            self._cols = ["V(vout)"]
            self._data = {"V(vout)": [3.14, 2.71]}

        @property
        def columns(self):
            return self._cols

        def select_dtypes(self, include=None):
            # return self-like with columns attr set to numeric column list
            return types.SimpleNamespace(columns=self._cols)

        def __getitem__(self, key):
            return self._data[key]

        def tolist(self):
            return self._data[self._cols[0]]

    samples = [{"R": 1000.0}, {"R": 1100.0}]
    fake_mc = FakeMC(samples, DF())

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

    from examples.monte_carlo_demo import _orchestrator_mc

    samples_out, metrics = _orchestrator_mc(2, "ngspice")
    assert len(samples_out) == len(samples)
    for out, expected in zip(samples_out, samples, strict=False):
        assert out["R"] == expected["R"]
        # default C value is filled in when absent
        assert "C" in out
    assert len(metrics) == 2
    assert abs(metrics[0] - 3.14) < 1e-6


def test_orchestrator_mc_fallback_no_numeric(monkeypatch):
    # DataFrame-like object with no numeric columns
    class DF:
        def __init__(self):
            self._cols = ["label"]
            self._data = {"label": ["a", "b"]}

        @property
        def columns(self):
            return self._cols

        def select_dtypes(self, include=None):
            return types.SimpleNamespace(columns=[])

        def __getitem__(self, key):
            return self._data[key]

        def tolist(self):
            return self._data[self._cols[0]]

    samples = [{"R": 1000.0}, {"R": 1100.0}]
    fake_mc = FakeMC(samples, DF())

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

    from examples.monte_carlo_demo import _orchestrator_mc

    samples_out, metrics = _orchestrator_mc(2, "ngspice")
    assert len(samples_out) == len(samples)
    assert len(metrics) == 2
