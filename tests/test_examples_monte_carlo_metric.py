import numpy as np


def _make_xarray_like(var_name: str, value: float):
    """Return a tiny xarray-like object with data_vars and mapping access."""

    class Var:
        def __init__(self, v):
            self.values = np.array([v])

    class DS:
        def __init__(self, name, v):
            self.data_vars = {name: Var(v)}

        def __getitem__(self, key):
            return self.data_vars[key]

        def keys(self):
            return list(self.data_vars.keys())

    return DS(var_name, value)


class FakeResultHandle:
    def __init__(self, ds):
        self._ds = ds

    def dataset(self):
        return self._ds


def test_real_mc_extracts_vout_metric(monkeypatch, tmp_path):
    # Import the function under test lazily to avoid side-effects at module import
    from examples.monte_carlo_demo import real_mc

    # Create a minimal xarray-like dataset where voltage variable is 'V(vout)'
    ds = _make_xarray_like("V(vout)", 3.1415)
    fake_handle = FakeResultHandle(ds)

    # Monkeypatch run_simulation to return our fake handle
    def _fake_run_simulation(circuit, analyses, engine=None, **kwargs):
        return fake_handle

    # patch the name used by the examples module (it imports run_simulation at module import)
    import examples.monte_carlo_demo as demo_mod

    monkeypatch.setattr(demo_mod, "run_simulation", _fake_run_simulation)

    samples, metrics = real_mc(1, engine="ngspice")
    assert len(samples) == 1
    assert len(metrics) == 1
    assert abs(metrics[0] - 3.1415) < 1e-6
