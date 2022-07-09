from global_settings import date0_min, date0_max
from params.params import window_dict, params_dict
from experiments.experiment import experiment
from experiments.generator import generate_window
from experiments.summary import summarize
from global_settings import OUTPUT_PATH
import json
import os


def run_experiment(model_name, horizons):
    """ Perform experiment for a particular model
    :param model_name: model name
    :param horizons: list of horizons
    """

    # make directory for the model
    model_path = os.path.join(OUTPUT_PATH, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # save parameter information
    params = params_dict[model_name]
    params_path = os.path.join(model_path, "params")
    if not os.path.isdir(params_path):
        os.mkdir(params_path)

    with open(os.path.join(params_path, "window.json"), "w") as handle:
        json.dump(window_dict, handle)

    with open(os.path.join(params_path, "params.json"), "w") as handle:
        json.dump(params, handle)

    # perform experiments
    for horizon in horizons:
        horizon_path = os.path.join(model_path, f"horizon={horizon}")
        if not os.path.isdir(horizon_path):
            os.mkdir(horizon_path)

        window_gen = list(generate_window(window_dict, date0_min, date0_max, horizon))
        for window in window_gen:
            experiment_proc(window, model_name, horizon, params)

        # # CPU: better to use a single process; GPU: better to use two processes
        # partial_func = functools.partial(experiment_proc, model_name=model_name, horizon=horizon, params=params)
        # pool = multiprocessing.Pool(2)  # number of processes
        # pool.map(partial_func, window_gen, chunksize=1)
        # pool.close()
        # pool.join()


def experiment_proc(window, model_name, horizon, params):
    """ Multi-processing for experiments
    :param window: [trddt_train, trddt_valid, trddt_test] window
    :param model_name: model name
    :param horizon: horizon for prediction
    :param params: dictionary of parameters
    """

    # make directory for the window
    model_path = os.path.join(OUTPUT_PATH, model_name)
    horizon_path = os.path.join(model_path, f"horizon={horizon}")
    window_path = os.path.join(horizon_path, window["X"][0][0])

    if not os.path.isdir(window_path):
        os.mkdir(window_path)
        os.mkdir(os.path.join(window_path, "info"))
        os.mkdir(os.path.join(window_path, "predict"))
        os.mkdir(os.path.join(window_path, "summary"))
        experiment(model_name, horizon, params, window)
        summarize(model_name, horizon, window)


if __name__ == "__main__":
    # model_name = "autogluon"
    # horizons = [1, 2, 3, 4, 5, 10, 20, 30, 50]
    # run_experiment(model_name, horizons)

    from experiments.correlation import plot_correlation
    model_name = "autogluon"
    horizons = 1
    plot_correlation(model_name, horizons)
