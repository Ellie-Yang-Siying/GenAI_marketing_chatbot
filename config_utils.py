import mlflow

def load_config(config_type, config_file='config.yaml'):
    """
    Load a section of the config.yaml file (e.g., 'model', 'mlflow', 'langgraph', etc.)

    Args:
        config_type (str): the top-level section of config.yaml to retrieve.
        config_file (str): the path to the yaml file (default: config.yaml)

    Returns:
        dict: the requested configuration block.
    """
    config = mlflow.models.ModelConfig(development_config=config_file)

    config_mapping = {
        "model": config.get("model"),
        "langgraph": config.get("langgraph"),
        "mlflow": config.get("mlflow"),
        "genie": config.get("genie")
    }

    config_types = config_mapping.keys()

    if config_type in config_types:
        return config_mapping[config_type]
    else:
        raise Exception(
            f"The config type '{config_type}' is not among the supported types: {', '.join(config_types)}"
        )
