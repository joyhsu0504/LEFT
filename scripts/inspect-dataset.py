#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : inspect-dataset.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/26/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.

import os.path as osp
import jacinle

g_dataset_loaders = {
    'clevr': 'load_CLEVR'
}

parser = jacinle.JacArgumentParser(usage="""
jac-run scripts/inspect-dataset.py clevr --data-dir ~/data/visual-concept/clevr-new
""".strip())
parser.add_argument('dataset', choices=g_dataset_loaders.keys())
parser.add_argument('--data-dir', type='checked_dir')
args = parser.parse_args()


def main():
    dataset = globals()[g_dataset_loaders[args.dataset]](args.data_dir)

    print('Dataset statistics:')
    print('  Length:', len(dataset))
    print('Dataset examples:')
    jacinle.stprint(dataset[0], 'dataset[0]', max_depth=1)

    from IPython import embed; embed()


def load_CLEVR(data_dir: str):
    from concepts.benchmark.clevr.dataset import make_dataset

    return make_dataset(
        scenes_json=osp.join(args.data_dir, 'scenes.json'),
        questions_json=osp.join(args.data_dir, 'questions.json'),
        image_root=osp.join(args.data_dir, 'images'),
        vocab_json=osp.join(args.data_dir, 'vocab.json')
    )


if __name__ == '__main__':
    main()

