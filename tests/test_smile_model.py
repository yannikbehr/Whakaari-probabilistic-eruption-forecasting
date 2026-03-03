from whakaaribn.smile_model import WhakaariSmileModel
from whakaaribn.model import WhakaariModel
from whakaaribn import Discretizer, split_by_group
from sklearn.pipeline import Pipeline
from sklearn import set_config
import numpy as np
import pytest

pytest.importorskip("pysmile", reason="pysmile is not installed")


set_config(transform_output="pandas")


def test_io(tmp_path):
    fout = tmp_path / "test_network.xdsl"
    ml = WhakaariModel(randomize=True, seed=42, modelfile=str(fout))
    npts = 100
    data = ml.simulate(n_samples=npts)
    ml1 = WhakaariSmileModel(modelfile=str(fout))
    ml1.from_pgmpy_model(ml.create_network())
    predict_data = data.drop("eruptions", axis=1)
    ml2 = WhakaariSmileModel(modelfile=str(fout))
    predictions = ml2.predict_proba(predict_data)
    assert (
        abs(
            predictions[:, 0].mean(
            ) - ml.create_network().get_cpds("eruptions").values[0]
        )
        < 0.1
    )
    assert (
        abs(
            predictions[:, 1].mean(
            ) - ml.create_network().get_cpds("eruptions").values[1]
        )
        < 0.1
    )


def test_fit():
    ml = WhakaariModel(randomize=True, seed=42)
    npts = 1000
    data = ml.simulate(n_samples=npts)
    train_data = data.iloc[: int(0.8 * npts)]
    eruptions = train_data["eruptions"]
    train_data = train_data.drop("eruptions", axis=1)
    predict_data = data.iloc[int(0.8 * npts):]
    ml1 = WhakaariSmileModel(uniformize=True)
    ml1.fit(train_data, eruptions)
    predict_data = predict_data.drop("eruptions", axis=1)
    predictions = ml1.predict_proba(predict_data)
    assert (
        abs(
            predictions[:, 0].mean(
            ) - ml.create_network().get_cpds("eruptions").values[0]
        )
        < 0.1
    )
    assert (
        abs(
            predictions[:, 1].mean(
            ) - ml.create_network().get_cpds("eruptions").values[1]
        )
        < 0.1
    )


def test_pipeline():
    pipe = Pipeline(
        [("discretize", Discretizer()),
         ("clf", WhakaariSmileModel(uniformize=True, smoothing=30))]
    )
    ml = WhakaariModel(randomize=True, seed=42)
    data = ml.simulate(n_samples=1000, mode="continuous")
    pipe.fit(
        data.drop("eruptions", axis=1).iloc[:800], data["eruptions"].iloc[:800])
    predictions = pipe.predict_proba(data.drop("eruptions", axis=1).iloc[800:])
    assert (
        abs(
            predictions[:, 0].mean(
            ) - ml.create_network().get_cpds("eruptions").values[0]
        )
        < 0.1
    )
    assert (
        abs(
            predictions[:, 1].mean(
            ) - ml.create_network().get_cpds("eruptions").values[1]
        )
        < 0.1
    )


def test_real_data(setup_real_data):
    data = setup_real_data
    pipe = Pipeline(
        [
            ("discretize", Discretizer()),
            ("clf", WhakaariSmileModel(smoothing=30, uniformize=True)),
        ]
    )
    pipe.set_output(transform="pandas")
    X_train, y_train, X_remainder, y_remainder = split_by_group(
        data, group="e")
    pipe.fit(X_train.ffill(), y_train)
    predictions = pipe.predict_proba(X_train.ffill())
    assert predictions.shape[0] == X_train.shape[0]
    assert np.isnan(predictions).sum() == 0
