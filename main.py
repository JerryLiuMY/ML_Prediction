from global_settings import date0_min, date0_max
from params.params import window_dict, params_dict, horizon_dict
from experiments.experiment import experiment
from experiments.generator import generate_window
from experiments.summary import summarize
from experiments.correlation import plot_correlation
from global_settings import OUTPUT_PATH
import multiprocessing
import functools
import json
import os


def run_experiment(model_name, num_proc):
    """ Perform experiment for a particular model
    :param model_name: model name
    :param num_proc: number of parallel processes
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
        with open(os.path.join(params_path, "horizon.json"), "w") as handle:
            json.dump(horizon_dict, handle)

    # perform experiments
    seq_len, horizon = params["seq_len"], horizon_dict["horizon"]
    window_gen = list(generate_window(window_dict, date0_min, date0_max, seq_len, horizon))

    if num_proc == 1:
        for window in window_gen:
            experiment_proc(window, model_name, horizon, params)
    else:
        partial_func = functools.partial(experiment_proc, model_name=model_name, horizon=horizon, params=params)
        pool = multiprocessing.Pool(num_proc)  # number of processes
        pool.map(partial_func, window_gen, chunksize=1)
        pool.close()
        pool.join()

    plot_correlation(model_name)


def experiment_proc(window, model_name, horizon, params):
    """ Multi-processing for experiments
    :param window: [trddt_train, trddt_valid, trddt_test] window
    :param model_name: model name
    :param horizon: horizon for prediction
    :param params: dictionary of parameters
    """

    # make directory for the window
    model_path = os.path.join(OUTPUT_PATH, model_name)
    window_path = os.path.join(model_path, window["name"])

    if not os.path.isdir(window_path):
        os.mkdir(window_path)
        os.mkdir(os.path.join(window_path, "info"))
        os.mkdir(os.path.join(window_path, "model"))
        os.mkdir(os.path.join(window_path, "predict"))
        os.mkdir(os.path.join(window_path, "summary"))
        experiment(model_name, horizon, params, window)
        summarize(model_name, window)


if __name__ == "__main__":
    run_experiment(model_name="transformer", num_proc=1)
