import numpy as np
from sklearn import set_config
from sklearn.pipeline import Pipeline

from whakaaribn import Discretizer, pre_eruption_window, split_by_group
from whakaaribn.model import WhakaariModel

set_config(transform_output="pandas")


def test_io(tmp_path):
    fout = tmp_path / "test_network.bif"
    ml = WhakaariModel(randomize=True, seed=42, modelfile=str(fout))
    npts = 100
    data = ml.simulate(n_samples=npts)
    ml1 = WhakaariModel(modelfile=str(fout))
    predict_data = data.drop("eruptions", axis=1)
    predictions = ml1.predict_proba(predict_data)
    assert (
        abs(
            predictions[:, 0].mean()
            - ml.create_network().get_cpds("eruptions").values[0]
        )
        < 0.1
    )
    assert (
        abs(
            predictions[:, 1].mean()
            - ml.create_network().get_cpds("eruptions").values[1]
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
    predict_data = data.iloc[int(0.8 * npts) :]
    ml1 = WhakaariModel(uniformize=True)
    ml1.fit(train_data, eruptions)
    predict_data = predict_data.drop("eruptions", axis=1)
    predictions = ml1.predict_proba(predict_data)
    assert (
        abs(
            predictions[:, 0].mean()
            - ml.create_network().get_cpds("eruptions").values[0]
        )
        < 0.1
    )
    assert (
        abs(
            predictions[:, 1].mean()
            - ml.create_network().get_cpds("eruptions").values[1]
        )
        < 0.1
    )


def test_pipeline():
    pipe = Pipeline(
        [
            ("discretize", Discretizer()),
            ("clf", WhakaariModel(uniformize=True, smoothing=30)),
        ]
    )
    ml = WhakaariModel(randomize=True, seed=42)
    data = ml.simulate(n_samples=1000, mode="continuous")
    pipe.fit(data.drop("eruptions", axis=1).iloc[:800], data["eruptions"].iloc[:800])
    predictions = pipe.predict_proba(data.drop("eruptions", axis=1).iloc[800:])
    assert (
        abs(
            predictions[:, 0].mean()
            - ml.create_network().get_cpds("eruptions").values[0]
        )
        < 0.1
    )
    assert (
        abs(
            predictions[:, 1].mean()
            - ml.create_network().get_cpds("eruptions").values[1]
        )
        < 0.1
    )


def test_bayesian_estimator():
    ml = WhakaariModel(randomize=True, seed=42)
    npts = 1000
    data = ml.simulate(n_samples=npts)
    gen = np.random.default_rng(seed=42)
    mask = gen.choice([True, False], size=data.shape, p=[0.1, 0.9])
    data_masked = data.mask(mask)
    train_data = data_masked.iloc[: int(0.8 * npts)]
    eruptions = train_data["eruptions"]
    train_data = train_data.drop("eruptions", axis=1)
    predict_data = data_masked.iloc[int(0.8 * npts) :]
    ml1 = WhakaariModel(uniformize=True)
    ml1.fit(train_data, eruptions, method="bayesian_estimation")
    predict_data = predict_data.drop("eruptions", axis=1)
    predictions = ml1.model.predict_probability(predict_data)
    assert predictions.shape[0] == predict_data.shape[0]
    assert np.isnan(predictions.values).sum() == 0


def test_real_data(setup_real_data):
    data = setup_real_data
    pipe = Pipeline(
        [
            ("discretize", Discretizer()),
            ("clf", WhakaariModel(smoothing=30, uniformize=True, pew=0)),
        ]
    )
    pipe.set_output(transform="pandas")
    X_train, y_train, X_remainder, y_remainder = split_by_group(data, group="e")
    _y_train = pre_eruption_window(y_train, 30)
    pipe.fit(X_train.ffill(), _y_train)
    predictions = pipe.predict_proba(X_train.ffill())
    assert predictions.shape[0] == X_train.shape[0]
    assert np.isnan(predictions).sum() == 0
    assert abs(predictions[:, 1].mean() - 0.03) < 0.001

    pipe1 = Pipeline(
        [
            ("discretize", Discretizer()),
            ("clf", WhakaariModel(smoothing=30, uniformize=True, pew=30)),
        ]
    )
    pipe1.set_output(transform="pandas")
    X_train, y_train, X_remainder, y_remainder = split_by_group(data, group="e")
    pipe1.fit(X_train.ffill(), y_train)
    predictions = pipe1.predict_proba(X_train.ffill())
    assert predictions.shape[0] == X_train.shape[0]
    assert np.isnan(predictions).sum() == 0
    assert abs(predictions[:, 1].mean() - 0.03) < 0.001
