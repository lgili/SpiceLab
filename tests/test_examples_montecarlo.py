from examples.monte_carlo_demo import run_demo


def test_montecarlo_demo_runs() -> None:
    df = run_demo(3)
    assert df is not None
    # Expect at least 3 rows (trials)
    assert len(df) == 3
