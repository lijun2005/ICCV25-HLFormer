import os
import yaml


cfg = {}



cfg["model_name"] = f"HLFormer"
cfg['dataset_name'] = 'tvr'
cfg['seed'] = 0
cfg['root'] = '/data2/lianniu/'
cfg['data_root'] = '/data2/lianniu/PRVR_data/PRVR'

cfg['visual_feature'] = 'i3d_resnet'
cfg['collection'] = 'tvr'
cfg['map_size'] = 32
cfg['clip_scale_w'] = 0.7
cfg['frame_scale_w'] = 0.3

cfg['model_root'] = os.path.join(cfg['root'], 'prvr_model', cfg['dataset_name'], cfg['model_name'])
cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')


# extra
cfg['sft_factor'] = 0.09


# dataset
cfg['num_workers'] = 32
cfg['no_core_driver'] = False
cfg['no_pin_memory'] = False
cfg['batchsize'] = 128


# opt
cfg['lr'] = 0.0003
cfg['lr_warmup_proportion'] = 0.01
cfg['wd'] = 0.01
cfg['margin'] = 0.1


# train
cfg['n_epoch'] = 100
cfg['max_es_cnt'] = 10
cfg['hard_negative_start_epoch'] = 20
cfg['hard_pool_size'] = 20
cfg['use_hard_negative'] = False
cfg['loss_factor'] = [0.05, 0.04, 8e-5, 1e-3]
cfg['neg_factor'] = [0.15, 32, 1]



cfg["curv_init"] = 1
cfg["learn_curv_bool"] = True
cfg["lorentz_dim"] = 127
cfg['attention_num'] = 8

# eval
cfg['eval_query_bsz'] = 50
cfg['eval_context_bsz'] = 100


# model
cfg['max_desc_l'] = 30
cfg['max_ctx_l'] = 128
cfg['sub_feat_size'] = 768
cfg['q_feat_size'] = 768
cfg['max_position_embeddings'] = 300
cfg['hidden_size'] = 384
cfg['n_heads'] = 4
cfg['input_drop'] = 0.2
cfg['drop'] = 0.2
cfg['initializer_range'] = 0.02


cfg['num_workers'] = 1 if cfg['no_core_driver'] else cfg['num_workers']
cfg['pin_memory'] = not cfg['no_pin_memory']


if not os.path.exists(cfg['model_root']):
    os.makedirs(cfg['model_root'], exist_ok=True)
if not os.path.exists(cfg['ckpt_path']):
    os.makedirs(cfg['ckpt_path'], exist_ok=True)


def get_cfg_defaults():
    with open(os.path.join(cfg['model_root'], 'hyperparams.yaml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    return cfg