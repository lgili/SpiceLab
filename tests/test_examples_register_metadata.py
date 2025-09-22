from pathlib import Path

from examples import register_and_metadata


def test_register_and_metadata_smoke(tmp_path: Path) -> None:
    # run the example; it writes a models file and runs OP. The example will use
    # a real ngspice runner if one is available in the environment. The example
    # unregisters itself; we just ensure it runs without exception and the
    # models file exists
    register_and_metadata.main(outdir=tmp_path)
    models_path = tmp_path / "models" / "custom_models.lib"
    assert models_path.exists()
    # cleanup the models file created by the example
    try:
        models_path.unlink()
    except Exception:
        pass
