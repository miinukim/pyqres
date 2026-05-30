from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import hydra
import numpy as np
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from pyqres.baselines import (
    ESNConfig,
    EchoStateNetwork,
    SoftmaxReadoutConfig,
    fit_softmax_readout,
    predict_softmax_readout,
    run_channel_equalization_esn,
    run_stm_esn,
)
from pyqres.exact.channel_map import ChannelMapReservoir, ChannelMapReservoirConfig
from pyqres.core.control import MeasurementControlConfig
from pyqres.exact.hardware import HardwareTrajectoryReservoir, HardwareTrajectoryReservoirConfig
from pyqres.core.reservoir_params import ReservoirParams
from pyqres.tasks import (
    ChannelEqualizationConfig,
    ChannelEqualizationDatasetConfig,
    ChannelEqualizationTaskRunner,
    STMConfig,
    STMTaskRunner,
    collect_channel_equalization_reservoir_features,
    generate_channel_equalization_data,
    generate_channel_equalization_dataset,
)


ReservoirFactory = Callable[[], Any]


def _make_output_dir() -> Path:
    return Path(HydraConfig.get().runtime.output_dir)


def _output_dir_str() -> str:
    if HydraConfig.initialized():
        return str(_make_output_dir())
    return "N/A"


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_builtin(payload), fh, indent=2, sort_keys=True)


def _save_arrays(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _save_run_artifacts(
    cfg: DictConfig,
    metrics: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    summary: Dict[str, Any],
) -> None:
    outdir = _make_output_dir()
    if bool(cfg.logging.save_resolved_config):
        OmegaConf.save(cfg, outdir / "resolved_config.yaml", resolve=True)
    if bool(cfg.logging.save_metrics):
        _save_json(outdir / "metrics.json", metrics)
    if bool(cfg.logging.save_summary):
        _save_json(outdir / "run_summary.json", summary)
    if bool(cfg.logging.save_arrays) and arrays:
        _save_arrays(outdir / "arrays" / "results.npz", arrays)


def _print_metrics(metrics: Dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def _nearest_symbol(values: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)[..., None]
    symbols = np.asarray(symbols, dtype=float)
    return symbols[np.argmin(np.abs(values - symbols[None, ...]), axis=-1)]


def _collect_esn_symbol_features(esn_cfg: ESNConfig, observed_messages: np.ndarray) -> np.ndarray:
    observed_messages = np.asarray(observed_messages, dtype=float)
    features = []
    for message in observed_messages:
        esn = EchoStateNetwork(esn_cfg)
        features.append(np.asarray(esn.collect_states(message.tolist()), dtype=float))
    return np.stack(features, axis=0)


def _evaluate_symbol_qrc(
    reservoir: ChannelMapReservoir,
    dataset: dict[str, np.ndarray],
    readout_cfg: SoftmaxReadoutConfig,
    initial_state: np.ndarray,
) -> dict[str, float]:
    train_features = collect_channel_equalization_reservoir_features(
        reservoir,
        dataset["train_observed"],
        initial_state=initial_state,
    )
    test_features = collect_channel_equalization_reservoir_features(
        reservoir,
        dataset["test_observed"],
        initial_state=initial_state,
    )
    X_train = train_features.reshape(-1, train_features.shape[-1])
    X_test = test_features.reshape(-1, test_features.shape[-1])
    y_train = dataset["train_messages"].reshape(-1)
    y_test = dataset["test_messages"].reshape(-1)

    model = fit_softmax_readout(X_train, y_train, readout_cfg)
    train_pred = predict_softmax_readout(X_train, model)
    test_pred = predict_softmax_readout(X_test, model)
    return {
        "train_error_rate": float(np.mean(train_pred != y_train)),
        "test_error_rate": float(np.mean(test_pred != y_test)),
    }


def _evaluate_symbol_esn(
    dataset: dict[str, np.ndarray],
    esn_cfg: ESNConfig,
    readout_cfg: SoftmaxReadoutConfig,
) -> dict[str, float]:
    train_features = _collect_esn_symbol_features(esn_cfg, dataset["train_observed"])
    test_features = _collect_esn_symbol_features(esn_cfg, dataset["test_observed"])
    X_train = train_features.reshape(-1, train_features.shape[-1])
    X_test = test_features.reshape(-1, test_features.shape[-1])
    y_train = dataset["train_messages"].reshape(-1)
    y_test = dataset["test_messages"].reshape(-1)

    model = fit_softmax_readout(X_train, y_train, readout_cfg)
    train_pred = predict_softmax_readout(X_train, model)
    test_pred = predict_softmax_readout(X_test, model)
    return {
        "train_error_rate": float(np.mean(train_pred != y_train)),
        "test_error_rate": float(np.mean(test_pred != y_test)),
    }


def _evaluate_symbol_logistic(
    dataset: dict[str, np.ndarray],
    readout_cfg: SoftmaxReadoutConfig,
) -> dict[str, float]:
    X_train = dataset["train_observed"].reshape(-1, 1)
    X_test = dataset["test_observed"].reshape(-1, 1)
    y_train = dataset["train_messages"].reshape(-1)
    y_test = dataset["test_messages"].reshape(-1)

    model = fit_softmax_readout(X_train, y_train, readout_cfg)
    train_pred = predict_softmax_readout(X_train, model)
    test_pred = predict_softmax_readout(X_test, model)
    return {
        "train_error_rate": float(np.mean(train_pred != y_train)),
        "test_error_rate": float(np.mean(test_pred != y_test)),
    }


def _evaluate_symbol_naive(dataset: dict[str, np.ndarray]) -> dict[str, float]:
    symbols = dataset["symbols"]
    train_pred = _nearest_symbol(dataset["train_observed"], symbols)
    test_pred = _nearest_symbol(dataset["test_observed"], symbols)
    return {
        "train_error_rate": float(np.mean(train_pred != dataset["train_messages"])),
        "test_error_rate": float(np.mean(test_pred != dataset["test_messages"])),
    }


def _plot_symbol_channel_equalization(
    outdir: Path,
    snr_list: np.ndarray,
    qrc_errors: np.ndarray,
    esn_errors: np.ndarray,
    logistic_errors: np.ndarray,
    naive_errors: np.ndarray,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(snr_list, naive_errors, "k-.", linewidth=2.5, label="Naive rounding")
    ax.plot(snr_list, logistic_errors, color="#bcbd22", marker="+", markersize=14, linewidth=2.5, label="Logistic")
    ax.plot(snr_list, esn_errors, color="#1f77b4", marker="s", markersize=8, linewidth=2.4, label="ESN")
    ax.plot(snr_list, qrc_errors, color="#d62728", marker="o", markersize=10, linewidth=2.8, label="QRC")
    ax.set_yscale("log")
    ax.set_xlim(float(np.min(snr_list)), float(np.max(snr_list)))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test Symbol Error Rate")
    ax.set_title("Channel Equalization")
    ax.grid(True, which="both", linestyle=":", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = outdir / "channel_equalization_snr.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def _build_reservoir_factory(cfg: DictConfig) -> ReservoirFactory:
    kind = str(cfg.reservoir.kind)

    control_cfg = MeasurementControlConfig(
        measurement_mode=str(cfg.reservoir.control.measurement_mode),
        measurement_strength=float(cfg.reservoir.control.measurement_strength),
        post_measurement_mode=str(cfg.reservoir.control.post_measurement_mode),
        conditioned_gate=str(cfg.reservoir.control.conditioned_gate),
        conditioned_gate_angle=float(cfg.reservoir.control.conditioned_gate_angle),
        conditioned_gate_target=int(cfg.reservoir.control.conditioned_gate_target),
        conditioned_gate_condition=str(cfg.reservoir.control.conditioned_gate_condition),
    )

    if kind == "channel_map":
        def factory() -> ChannelMapReservoir:
            rcfg = ChannelMapReservoirConfig(
                n_system=int(cfg.reservoir.n_system),
                n_ancilla=int(cfg.reservoir.n_ancilla),
                tau=float(cfg.reservoir.tau),
                input_scale=float(cfg.reservoir.input_scale),
                include_bias=bool(cfg.reservoir.include_bias),
                use_shot_noise=bool(cfg.reservoir.use_shot_noise),
                shots=int(cfg.reservoir.shots),
                init_state=str(cfg.reservoir.init_state),
                seed=int(cfg.reservoir.seed),
                control=control_cfg,
            )
            return ChannelMapReservoir(rcfg)

        return factory

    if kind == "hardware_trajectory":
        def factory() -> HardwareTrajectoryReservoir:
            rcfg = HardwareTrajectoryReservoirConfig(
                n_system=int(cfg.reservoir.n_system),
                n_ancilla=int(cfg.reservoir.n_ancilla),
                tau=float(cfg.reservoir.tau),
                input_scale=float(cfg.reservoir.input_scale),
                include_bias=bool(cfg.reservoir.include_bias),
                init_state=str(cfg.reservoir.init_state),
                shots=int(cfg.reservoir.shots),
                seed=int(cfg.reservoir.seed),
                control=control_cfg,
            )
            return HardwareTrajectoryReservoir(rcfg)

        return factory

    raise ValueError(f"Unsupported reservoir kind: {kind}")


def _esn_config(cfg: DictConfig) -> ESNConfig:
    return ESNConfig(
        n_res=int(cfg.baseline.n_res),
        spectral_radius=float(cfg.baseline.spectral_radius),
        input_scale=float(cfg.baseline.input_scale),
        leak_rate=float(cfg.baseline.leak_rate),
        ridge_l2=float(cfg.baseline.ridge_l2),
        seed=int(cfg.baseline.seed),
        state_clip=float(cfg.baseline.state_clip),
        power_iter=int(cfg.baseline.power_iter),
    )


def _run_stm(cfg: DictConfig, reservoir_factory: ReservoirFactory) -> tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    task_cfg = STMConfig(
        T_total=int(cfg.task.T_total),
        washout=int(cfg.task.washout),
        train_len=int(cfg.task.train_len),
        test_len=int(cfg.task.test_len),
        delays=tuple(int(d) for d in cfg.task.delays),
        input_dist=str(cfg.task.input_dist),
        input_seed=int(cfg.task.input_seed),
        ridge_l2=float(cfg.task.ridge_l2),
        metric=str(cfg.task.metric),
    )

    analysis_runner = STMTaskRunner(reservoir_factory(), task_cfg)
    inputs = analysis_runner.generate_inputs()
    features = reservoir_factory().run_stream(inputs.tolist())

    quantum_runner = STMTaskRunner(reservoir_factory(), task_cfg)
    qres = quantum_runner.run()
    quantum_mc = quantum_runner.memory_capacity(qres, use_test=True)

    esn_cfg = _esn_config(cfg)
    eres = run_stm_esn(
        T_total=task_cfg.T_total,
        washout=task_cfg.washout,
        train_len=task_cfg.train_len,
        test_len=task_cfg.test_len,
        delays=task_cfg.delays,
        input_dist=task_cfg.input_dist,
        input_seed=task_cfg.input_seed,
        esn_cfg=esn_cfg,
        metric=task_cfg.metric,
    )
    esn_mc = quantum_runner.memory_capacity(eres, use_test=True)

    delays = np.asarray(task_cfg.delays, dtype=int)
    q_train = np.asarray([qres[d]["train_score"] for d in task_cfg.delays], dtype=float)
    q_test = np.asarray([qres[d]["test_score"] for d in task_cfg.delays], dtype=float)
    e_train = np.asarray([eres[d]["train_score"] for d in task_cfg.delays], dtype=float)
    e_test = np.asarray([eres[d]["test_score"] for d in task_cfg.delays], dtype=float)

    metrics = {
        "task": "stm",
        "reservoir_kind": str(cfg.reservoir.kind),
        "quantum_memory_capacity": quantum_mc,
        "esn_memory_capacity": esn_mc,
        "quantum_best_test_score": float(np.max(q_test)),
        "esn_best_test_score": float(np.max(e_test)),
    }
    arrays = {
        "inputs": inputs.astype(float),
        "reservoir_features": np.asarray(features, dtype=float),
        "delays": delays,
        "quantum_train_scores": q_train,
        "quantum_test_scores": q_test,
        "esn_train_scores": e_train,
        "esn_test_scores": e_test,
    }
    summary = {
        "task": "stm",
        "metric": task_cfg.metric,
        "n_delays": len(task_cfg.delays),
        "output_dir": _output_dir_str(),
    }
    return metrics, arrays, summary


def _run_channel_equalization(
    cfg: DictConfig,
    reservoir_factory: ReservoirFactory,
) -> tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    task_cfg = ChannelEqualizationConfig(
        T_total=int(cfg.task.T_total),
        washout=int(cfg.task.washout),
        train_len=int(cfg.task.train_len),
        test_len=int(cfg.task.test_len),
        delay=int(cfg.task.delay),
        input_seed=int(cfg.task.input_seed),
        ridge_l2=float(cfg.task.ridge_l2),
        taps=tuple(float(v) for v in cfg.task.taps),
        nonlin2=float(cfg.task.nonlin2),
        nonlin3=float(cfg.task.nonlin3),
        noise_std=float(cfg.task.noise_std),
        metric=str(cfg.task.metric),
    )

    observed, target = generate_channel_equalization_data(task_cfg)
    features = reservoir_factory().run_stream(observed.tolist())

    quantum_runner = ChannelEqualizationTaskRunner(reservoir_factory(), task_cfg)
    qres = quantum_runner.run()

    esn_cfg = _esn_config(cfg)
    eres = run_channel_equalization_esn(
        observed=observed,
        target=target,
        washout=task_cfg.washout,
        train_len=task_cfg.train_len,
        test_len=task_cfg.test_len,
        esn_cfg=esn_cfg,
        ridge_l2=task_cfg.ridge_l2,
        metric=task_cfg.metric,
    )

    metrics = {
        "task": "channel_equalization",
        "reservoir_kind": str(cfg.reservoir.kind),
        "quantum_test_score": float(qres["test_score"]),
        "quantum_test_ber": float(qres["test_ber"]),
        "quantum_test_mse": float(qres["test_mse"]),
        "esn_test_score": float(eres["test_score"]),
        "esn_test_ber": float(eres["test_ber"]),
        "esn_test_mse": float(eres["test_mse"]),
    }
    arrays = {
        "observed": observed.astype(float),
        "target": target.astype(float),
        "reservoir_features": np.asarray(features, dtype=float),
    }
    summary = {
        "task": "channel_equalization",
        "metric": task_cfg.metric,
        "delay": task_cfg.delay,
        "output_dir": _output_dir_str(),
    }
    return metrics, arrays, summary


def run_experiment_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    reservoir_factory = _build_reservoir_factory(cfg)
    task_name = str(cfg.task.name)

    if task_name == "stm":
        metrics, arrays, summary = _run_stm(cfg, reservoir_factory)
    elif task_name == "channel_equalization":
        metrics, arrays, summary = _run_channel_equalization(cfg, reservoir_factory)
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    _save_run_artifacts(cfg, metrics=metrics, arrays=arrays, summary=summary)
    _print_metrics(metrics)
    print(f"output_dir: {_make_output_dir()}")
    return metrics


def stm_demo() -> None:
    cfg = OmegaConf.create(
        {
            "reservoir": {
                "kind": "channel_map",
                "n_system": 4,
                "n_ancilla": 2,
                "tau": 1.0,
                "input_scale": 1.0,
                "include_bias": True,
                "use_shot_noise": False,
                "shots": 4096,
                "init_state": "maximally_mixed",
                "seed": 17462,
                "control": {
                    "measurement_mode": "projective",
                    "measurement_strength": 1.0,
                    "post_measurement_mode": "reset",
                    "conditioned_gate": "none",
                    "conditioned_gate_angle": float(np.pi),
                    "conditioned_gate_target": 0,
                    "conditioned_gate_condition": "nonzero",
                },
            },
            "task": {
                "name": "stm",
                "T_total": 400,
                "washout": 50,
                "train_len": 200,
                "test_len": 100,
                "delays": list(range(1, 11)),
                "input_dist": "uniform_pm1",
                "input_seed": 2026,
                "ridge_l2": 1e-6,
                "metric": "r2",
            },
            "baseline": {
                "n_res": 200,
                "spectral_radius": 0.8,
                "input_scale": 0.1,
                "leak_rate": 0.3,
                "ridge_l2": 1e-4,
                "seed": 2,
                "state_clip": 5.0,
                "power_iter": 200,
            },
        }
    )
    metrics, _, _ = _run_stm(cfg, _build_reservoir_factory(cfg))
    _print_metrics(metrics)


def run_symbol_channel_equalization_benchmark_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    reservoir_params = ReservoirParams(
        n_system=int(cfg.reservoir_params.n_system),
        n_ancilla=int(cfg.reservoir_params.n_ancilla),
        tau=float(cfg.reservoir_params.tau),
        seed=int(cfg.reservoir_params.seed),
        hx0_base=float(cfg.reservoir_params.hx0_base),
        hz1_base=float(cfg.reservoir_params.hz1_base),
        hx0_std=float(cfg.reservoir_params.hx0_std),
        hz1_std=float(cfg.reservoir_params.hz1_std),
        hx0_scale=float(cfg.reservoir_params.hx0_scale),
        hz1_scale=float(cfg.reservoir_params.hz1_scale),
        J_scale=float(cfg.reservoir_params.J_scale),
        graph_kind=str(cfg.reservoir_params.graph_kind),
    ).generate()
    reservoir = ChannelMapReservoir(
        ChannelMapReservoirConfig(
            n_system=int(cfg.reservoir.n_system),
            n_ancilla=int(cfg.reservoir.n_ancilla),
            tau=float(cfg.reservoir.tau),
            input_scale=float(cfg.reservoir.input_scale),
            include_bias=bool(cfg.reservoir.include_bias),
            use_shot_noise=bool(cfg.reservoir.use_shot_noise),
            shots=int(cfg.reservoir.shots),
            init_state=str(cfg.reservoir.init_state),
            hx0_vec=reservoir_params["hx0_vec"],
            hz1_vec=reservoir_params["hz1_vec"],
            J_mat=reservoir_params["J_mat"],
            seed=int(cfg.reservoir.seed),
            control=MeasurementControlConfig(
                measurement_mode=str(cfg.reservoir.control.measurement_mode),
                measurement_strength=float(cfg.reservoir.control.measurement_strength),
                post_measurement_mode=str(cfg.reservoir.control.post_measurement_mode),
                conditioned_gate=str(cfg.reservoir.control.conditioned_gate),
                conditioned_gate_angle=float(cfg.reservoir.control.conditioned_gate_angle),
                conditioned_gate_target=int(cfg.reservoir.control.conditioned_gate_target),
                conditioned_gate_condition=str(cfg.reservoir.control.conditioned_gate_condition),
            ),
        )
    )
    rho_fp = reservoir.fixed_point()
    esn_cfg = ESNConfig(
        n_res=int(cfg.baseline.n_res),
        spectral_radius=float(cfg.baseline.spectral_radius),
        input_scale=float(cfg.baseline.input_scale),
        leak_rate=float(cfg.baseline.leak_rate),
        ridge_l2=float(cfg.baseline.ridge_l2),
        seed=int(cfg.baseline.seed),
        state_clip=float(cfg.baseline.state_clip),
        power_iter=int(cfg.baseline.power_iter),
    )
    readout_cfg = SoftmaxReadoutConfig(
        fit_intercept=bool(cfg.readout.fit_intercept),
        l2=float(cfg.readout.l2),
        max_iter=int(cfg.readout.max_iter),
        tol=float(cfg.readout.tol),
    )
    snr_list = np.asarray([float(x) for x in cfg.benchmark.snr_list], dtype=float)

    rows = []
    print("Channel equalization symbol benchmark")
    print("SNR(dB) | QRC test SER | ESN test SER | Logistic test SER | Naive test SER")
    print("-" * 78)
    for snr_db in snr_list:
        dataset_cfg = ChannelEqualizationDatasetConfig(
            n_train=int(cfg.dataset.n_train),
            n_test=int(cfg.dataset.n_test),
            n_symb=int(cfg.dataset.n_symb),
            snr_db=float(snr_db),
            input_seed=int(cfg.dataset.input_seed),
            symbols=tuple(float(x) for x in cfg.dataset.symbols),
            taps=tuple(float(x) for x in cfg.dataset.taps),
            nonlin2=float(cfg.dataset.nonlin2),
            nonlin3=float(cfg.dataset.nonlin3),
        )
        dataset = generate_channel_equalization_dataset(dataset_cfg)
        qout = _evaluate_symbol_qrc(reservoir, dataset, readout_cfg, initial_state=rho_fp)
        eout = _evaluate_symbol_esn(dataset, esn_cfg, readout_cfg)
        lout = _evaluate_symbol_logistic(dataset, readout_cfg)
        nout = _evaluate_symbol_naive(dataset)
        row = {
            "snr_db": float(snr_db),
            "qrc_train_ser": float(qout["train_error_rate"]),
            "qrc_test_ser": float(qout["test_error_rate"]),
            "esn_train_ser": float(eout["train_error_rate"]),
            "esn_test_ser": float(eout["test_error_rate"]),
            "logistic_train_ser": float(lout["train_error_rate"]),
            "logistic_test_ser": float(lout["test_error_rate"]),
            "naive_train_ser": float(nout["train_error_rate"]),
            "naive_test_ser": float(nout["test_error_rate"]),
        }
        rows.append(row)
        print(
            f"{snr_db:6.1f} | "
            f"{row['qrc_test_ser']:.6f} | "
            f"{row['esn_test_ser']:.6f} | "
            f"{row['logistic_test_ser']:.6f} | "
            f"{row['naive_test_ser']:.6f}"
        )

    outdir = _make_output_dir()
    outdir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, outdir / "resolved_config.yaml", resolve=True)
    rows_payload = {"rows": rows}
    _save_json(outdir / "results.json", rows_payload)
    qrc_errors = np.asarray([row["qrc_test_ser"] for row in rows], dtype=float)
    esn_errors = np.asarray([row["esn_test_ser"] for row in rows], dtype=float)
    logistic_errors = np.asarray([row["logistic_test_ser"] for row in rows], dtype=float)
    naive_errors = np.asarray([row["naive_test_ser"] for row in rows], dtype=float)
    plot_path = _plot_symbol_channel_equalization(
        outdir=outdir,
        snr_list=snr_list,
        qrc_errors=qrc_errors,
        esn_errors=esn_errors,
        logistic_errors=logistic_errors,
        naive_errors=naive_errors,
    )
    arrays = {
        "snr_db": snr_list,
        "qrc_test_ser": qrc_errors,
        "esn_test_ser": esn_errors,
        "logistic_test_ser": logistic_errors,
        "naive_test_ser": naive_errors,
    }
    _save_arrays(outdir / "arrays" / "benchmark_curves.npz", arrays)
    metrics = {
        "task": "channel_equalization_symbol_benchmark",
        "qrc_mean_test_ser": float(np.mean(qrc_errors)),
        "esn_mean_test_ser": float(np.mean(esn_errors)),
        "logistic_mean_test_ser": float(np.mean(logistic_errors)),
        "naive_mean_test_ser": float(np.mean(naive_errors)),
        "best_qrc_test_ser": float(np.min(qrc_errors)),
        "best_esn_test_ser": float(np.min(esn_errors)),
        "best_logistic_test_ser": float(np.min(logistic_errors)),
        "best_naive_test_ser": float(np.min(naive_errors)),
        "plot_path": str(plot_path),
    }
    _save_json(outdir / "metrics.json", metrics)
    _print_metrics(metrics)
    print(f"output_dir: {outdir}")
    return metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    run_experiment_from_cfg(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def stm_hydra(cfg: DictConfig) -> None:
    run_experiment_from_cfg(cfg)


@hydra.main(version_base=None, config_path="conf", config_name="channel_equalization_benchmark")
def channel_equalization_benchmark(cfg: DictConfig) -> None:
    run_symbol_channel_equalization_benchmark_from_cfg(cfg)
