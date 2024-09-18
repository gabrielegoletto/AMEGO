import logging

import torch

logger = logging.getLogger("global")


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    if len(missing_keys) > 0:
        logger.info("[Warning] missing keys: {}".format(missing_keys))
        logger.info("missing keys:{}".format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info(
            "[Warning] unused_pretrained_keys: {}".format(unused_pretrained_keys)
        )
        logger.info("unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    logger.info("used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters share common prefix 'module.'"""
    logger.info("remove prefix '{}'".format(prefix))

    def get_value(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    # f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {get_value(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info("load pretrained model from {}".format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path)

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")

    try:
        check_keys(model, pretrained_dict)
    except BaseException:
        logger.info(
            '[Warning]: using pretrain as features. Adding "features." as prefix'
        )
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = "features." + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    logger.info("restore from {}".format(ckpt_path))
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt["epoch"]
    best_acc = ckpt["best_acc"]
    arch = ckpt["arch"]
    ckpt_model_dict = remove_prefix(ckpt["state_dict"], "module.")
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt["optimizer"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return model, optimizer, epoch, best_acc, arch
