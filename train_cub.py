from datetime import datetime
import random
import os
import shutil
import pickle

import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse

from models.model_cub import CoOp
from datasets import load_data


def parse_option():
    parser = argparse.ArgumentParser('cross-domain ZSL with CLIP', add_help=False)

    parser.add_argument("--db_name", type=str, default='CUB', help='dataset name')
    parser.add_argument("--seed", type=int, default=1234, help='random seed')

    # datasets setting
    parser.add_argument("--datasets_path", type=str, 
                        default="/media/haifeng/88ee5152-40be-4551-8ce0-77751544a78f/projects/RuiChen-cdzsl/cdZSL/datasets/CUB_200_2011/CUB_200_2011", 
                        help='path to datasets')
    parser.add_argument("--data_split_path", type=str, 
                        default="/media/haifeng/88ee5152-40be-4551-8ce0-77751544a78f/projects/RuiChen-cdzsl/cdZSL/data/CUB", 
                        help='path to mat file for spliting datasets')
    parser.add_argument("--train_type", type=str, 
                        default="nature", 
                        help='style for training')
    parser.add_argument("--val_type", type=str, 
                        default="cartoon", 
                        help='style for validation')

    # model setting
    parser.add_argument("--arch", type=str, default='RN101', help='arch')
    parser.add_argument("--ckpt_path", type=str, 
                        default='/media/haifeng/88ee5152-40be-4551-8ce0-77751544a78f/projects/RuiChen-cdzsl/cdZSL/RN101.pt', 
                        help='ckpt_path')

    # optimization setting
    parser.add_argument("--lr", type=float, default=5e-1, help='num_runs')
    parser.add_argument("--wd", type=float, default=0.0, help='num_runs')
    parser.add_argument("--resolution", type=int, default=224, help='num_runs')
    parser.add_argument("--epochs", type=int, default=100, help='num_runs')
    parser.add_argument("--train_batch", type=int, default=128, help='num_runs')
    parser.add_argument("--val_batch", type=int, default=256, help='num_runs')

    # model setting
    parser.add_argument("--model", type=str, default='coop', help='model')
    parser.add_argument("--n_ctx", type=int, default=36, help='num_runs')
    # parser.add_argument("--prompt_bsz", type=int, default=4, help='num_runs')

    args, _ = parser.parse_known_args()

    # now = datetime.now()
    # timestamp = now.strftime("%Y%m%d%H%M%S")
    # args.save_path = './runs/' + args.db_name + '/' + timestamp

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(args.seed)
    print(args.arch)

    if args.db_name == 'CUB':
        train_ds, test_zsl, test_seen_ds, test_unseen_ds, attr, seenclasses_name, unseenclasses_name, \
            attribute_zsl, attribute_seen, attribute_gzsl, label_attr_sentence, test_seen_label = load_data.CUB_dataloader(args=args)
    elif args.db_name == 'SUN':
        train_ds, test_zsl, test_seen_ds, test_unseen_ds, attr, seenclasses_name, unseenclasses_name, \
            attribute_zsl, attribute_seen, attribute_gzsl, label_attr_sentence, test_seen_label = load_data.SUN_dataloader(args=args)
    elif args.db_name == 'AWA2':
        train_ds, test_zsl, test_seen_ds, test_unseen_ds, attr, seenclasses_name, unseenclasses_name, \
            attribute_zsl, attribute_seen, attribute_gzsl, label_attr_sentence, test_seen_label = load_data.AWA2_dataloader(args=args)

    train_dl = DataLoader(train_ds, batch_size=args.train_batch, num_workers=0, shuffle=True, drop_last=True)
    test_zsl_dl = DataLoader(test_zsl, batch_size=args.val_batch, num_workers=0, shuffle=False)
    test_seen_dl = DataLoader(test_seen_ds, batch_size=args.val_batch, num_workers=0, shuffle=False)
    test_unseen_dl = DataLoader(test_unseen_ds, batch_size=args.val_batch, num_workers=0, shuffle=False)

    model = CoOp(args, attr, seenclasses_name, unseenclasses_name, label_attr_sentence, len(attribute_gzsl), test_seen_label)

    model.fit(train_dl, test_zsl_dl, test_seen_dl, test_unseen_dl, attribute_seen, attribute_zsl, attribute_gzsl)
    print('finish fit')


if __name__ == '__main__':
    args = parse_option()
    main(args)
