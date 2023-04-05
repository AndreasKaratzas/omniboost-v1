
import gymnasium as gym
import copy


def preprocess_env(env, preprocessor_configs):
    """Apply a series of pre-processes to an 
    environment, before it is used by an agent.

    Parameters
    ----------
    env : gym.Env
        An environment.
    preprocessor_configs : List
        A list of preprocessor configs

    Returns
    -------
    gym.Env
        A preprocessed copy of the environment.
    """
    for preprocessor_config in preprocessor_configs:
        if "method" in preprocessor_config:
            try:
                preprocessor = getattr(
                    env.unwrapped, preprocessor_config["method"])
                if "args" in preprocessor_config:
                    env = preprocessor(preprocessor_config["args"])
                else:
                    env = preprocessor()
            except AttributeError:
                print("The environment does not have a {} method".format(
                    preprocessor_config["method"]))
        else:
            print("The method is not specified in ",
                         preprocessor_config)
    return env


def safe_deepcopy_env(obj):
    """Perform a deep copy of an environment 
    but without copying its viewer.

    Parameters
    ----------
    obj : gym.Env
        The environment as defined in by
        OpenAI Gym standards.

    Returns
    -------
    gym.Env
        An "unwrapped" version of the environment.
    """
    cls = obj.__class__
    result = cls.__new__(cls)
    memo = {id(obj): result}
    for k, v in obj.__dict__.items():
        if k not in ['viewer', '_monitor', 'grid_render']:
            if isinstance(v, gym.Env):
                setattr(result, k, safe_deepcopy_env(v))
            else:
                setattr(result, k, copy.deepcopy(v, memo=memo))
        else:
            setattr(result, k, None)
    return result
