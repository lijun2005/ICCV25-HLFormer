def get_configs(dataset_name):
    if dataset_name in ['tvr']:
        import Configs.tvr as tvr
        return tvr.get_cfg_defaults()
    elif dataset_name in ['act']:
        import Configs.act as act
        return act.get_cfg_defaults()
    elif dataset_name in ['cha']:
        import Configs.cha as cha
        return cha.get_cfg_defaults()