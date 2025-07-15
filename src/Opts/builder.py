from Opts.optimization import BertAdam

def get_opts(cfg, model, train_loader):

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    num_train_optimization_steps = len(train_loader) * cfg['n_epoch']
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=cfg['lr'],
                         weight_decay=cfg['wd'],
                         warmup=cfg['lr_warmup_proportion'],
                         t_total=num_train_optimization_steps,
                         schedule="warmup_linear")
    return optimizer