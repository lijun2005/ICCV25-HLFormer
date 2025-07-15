import os
import ipdb
from torch.utils.data import DataLoader

from Utils.basic_utils import BigFile, read_dict
from Datasets.data_provider import Dataset4PRVR, VisDataSet4PRVR, TxtDataSet4PRVR, \
                    collate_train, collate_frame_val, collate_text_val, read_video_ids


def get_datasets(cfg):

    rootpath = cfg['data_root']
    collection = cfg['collection']

    trainCollection = '%strain' % collection
    valCollection = '%sval' % collection

    cap_file = {
        'train': '%s.caption.txt' % trainCollection,
        'val': '%s.caption.txt' % valCollection,
    }

    text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)

    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x]) for x in cap_file}
    # Load visual features
    visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])

    visual_feats = BigFile(visual_feat_path)
    cfg['visual_feat_dim'] = visual_feats.ndims

    video2frames = read_dict(os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))

    train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg, video2frames=video2frames)

    val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg)

    val_video_ids_list = read_video_ids(caption_files['val'])
    val_video_dataset = VisDataSet4PRVR(visual_feats, video2frames, cfg, video_ids=val_video_ids_list)


    testCollection = '%stest' % collection
    test_cap_file = {'test': '%s.caption.txt' % testCollection}

    # caption
    test_caption_files = {x: os.path.join(rootpath, collection, 'TextData', test_cap_file[x])
                     for x in test_cap_file}

    test_text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    # Load visual features
    test_visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])

    test_visual_feats = BigFile(test_visual_feat_path)
    test_video2frames =  read_dict(os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))


    test_video_ids_list = read_video_ids(test_caption_files['test'])
    test_vid_dataset = VisDataSet4PRVR(test_visual_feats, test_video2frames, cfg, video_ids=test_video_ids_list)
    test_text_dataset = TxtDataSet4PRVR(test_caption_files['test'], test_text_feat_path, cfg)


    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['batchsize'],
                              shuffle=True,
                              pin_memory=cfg['pin_memory'],
                              num_workers=cfg['num_workers'],
                              collate_fn=collate_train)
    context_dataloader = DataLoader(val_video_dataset,
                                    collate_fn=collate_frame_val,
                                    batch_size=cfg['eval_context_bsz'],
                                    num_workers=cfg['num_workers'],
                                    shuffle=False,
                                    pin_memory=cfg['pin_memory'])
    query_eval_loader = DataLoader(val_text_dataset,
                                   collate_fn=collate_text_val,
                                   batch_size=cfg['eval_query_bsz'],
                                   num_workers=cfg['num_workers'],
                                   shuffle=False,
                                   pin_memory=cfg['pin_memory'])
    test_context_dataloader = DataLoader(test_vid_dataset,
                                    collate_fn=collate_frame_val,
                                    batch_size=cfg['eval_context_bsz'],
                                    num_workers=cfg['num_workers'],
                                    shuffle=False,
                                    pin_memory=cfg['pin_memory'])
    test_query_eval_loader = DataLoader(test_text_dataset,
                                   collate_fn=collate_text_val,
                                   batch_size=cfg['eval_query_bsz'],
                                   num_workers=cfg['num_workers'],
                                   shuffle=False,
                                   pin_memory=cfg['pin_memory'])

    return cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader