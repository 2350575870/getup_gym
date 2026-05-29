from getup_gym.HhdRslRl.basement.base_config.baseconfig import BaseConfig


def check_env_batch_config(env_cfg):
    if not hasattr(env_cfg, "TrainBatch"):
        print("No TrainBatch in env_cfg")
        return 0

    if not hasattr(env_cfg.TrainBatch, "batch_paramters"):
        print("No batch_paramters in env_cfg.TrainBatch")
        return 0

    return env_cfg.TrainBatch.batch_len


def update_class_from_dict_idx(obj, params, idx):
    for key, val in params.items():
        attr = getattr(obj, key, None)
        if (
            isinstance(attr, float)
            or isinstance(attr, int)
            or isinstance(attr, str)
            or isinstance(attr, bool)
            or isinstance(attr, list)
            or isinstance(attr, dict)
            or isinstance(attr, tuple)
        ):
            setattr(obj, key, val[idx])
        else:
            update_class_from_dict_idx(attr, val, idx)
    return


def parse_env_batch_config(env_cfg, idx: int):
    if not hasattr(env_cfg, "TrainBatch"):
        print("No TrainBatch in env_cfg")
        return env_cfg

    if not hasattr(env_cfg.TrainBatch, "batch_paramters"):
        print("No batch_paramters in env_cfg.TrainBatch")
        return env_cfg

    update_class_from_dict_idx(env_cfg, env_cfg.TrainBatch.batch_paramters, idx)

    return env_cfg


def parse_train_batch_config(train_cfg, idx: int):
    if not hasattr(train_cfg, "TrainBatch"):
        print("No TrainBatch in train_cfg")
        return train_cfg  # TODO: check if this is correct

    if not hasattr(train_cfg.TrainBatch, "batch_paramters"):
        print("No batch_paramters in train_cfg.TrainBatch")
        return train_cfg

    for key, value in train_cfg.TrainBatch.batch_paramters.items():
        if isinstance(value, list):
            setattr(train_cfg, key, value[idx])
        else:
            setattr(train_cfg, key, value)

    return train_cfg
