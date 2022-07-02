

def save_model(model, model_name, path):
    """ Save the trained model
    :param model: trained model
    :param model_name: model name
    :param path: save path
    :return:
    """

    if model_name == "autogluon":
        model.save(path)
