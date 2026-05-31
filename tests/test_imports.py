def test_core_imports():
    from pyqres.core import QRCReservoirProtocol, ReservoirRunResult, ReservoirStepResult

    assert QRCReservoirProtocol is not None
    assert ReservoirRunResult is not None
    assert ReservoirStepResult is not None


def test_compatibility_imports():
    from pyqres.simulation import ExactQRCModelConfig
    from pyqres.qiskit import QRCReservoir
    from pyqres.dim import IsingReservoirModel, QRCLibExactReservoirModel
    from pyqres.tasks import STMConfig
    from pyqres.baselines import ESNConfig

    assert ExactQRCModelConfig is not None
    assert QRCReservoir is not None
    assert IsingReservoirModel is not None
    assert QRCLibExactReservoirModel is not None
    assert STMConfig is not None
    assert ESNConfig is not None


def test_simulation_and_dim_smoke():
    import numpy as np

    from pyqres.dim import QRCLibExactReservoirModel
    from pyqres.simulation import ChannelMapReservoir, ChannelMapReservoirConfig

    cfg = ChannelMapReservoirConfig(n_system=1, n_ancilla=1, seed=1)
    reservoir = ChannelMapReservoir(cfg)
    assert reservoir.run(np.array([0.0, 0.1])).shape == (2, 3)

    model = QRCLibExactReservoirModel(config=cfg)
    assert model.ptm(0.0).shape == (4, 4)
