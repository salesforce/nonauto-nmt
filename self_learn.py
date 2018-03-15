# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
import torch
import numpy as np
from torchtext import data
from torchtext import datasets
from torch.nn import functional as F
from torch.autograd import Variable

import revtok
import logging
import random
import string
import traceback
import math
import uuid
import argparse
import os
import copy
import time

from tqdm import tqdm, trange
from model import Transformer, FastTransformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset
from utils import Metrics, Best, computeGLEU, computeBLEU, Cache, Batch, masked_sort, unsorted, computeGroupBLEU
from time import gmtime, strftime

import sys
from traceback import extract_tb
from code import interact
def interactive_exception(e_class, e_value, tb):
    sys.__excepthook__(e_class, e_value, tb)
    tb_stack = extract_tb(tb)
    locals_stack = []
    while tb is not None:
        locals_stack.append(tb.tb_frame.f_locals)
        tb = tb.tb_next
    while len(tb_stack) > 0:
        frame = tb_stack.pop()
        ls = locals_stack.pop()
        print('\nInterpreter at file "{}", line {}, in {}:'.format(
            frame.filename, frame.lineno, frame.name))
        print('  {}'.format(frame.line.strip()))
        interact(local=ls)
#sys.excepthook = interactive_exception

# check dirs
for d in ['models', 'runs', 'logs']:
    if not os.path.exists('./{}'.format(d)):
        os.mkdir('./{}'.format(d))

# params
parser = argparse.ArgumentParser(description='Train a Transformer model.')

# data
parser.add_argument('--data_prefix', type=str, default='../data/')
parser.add_argument('--dataset', type=str, default='iwslt', help='"flickr" or "iwslt"')
parser.add_argument('--language', type=str, default='ende', help='a combination of two language markers to show the language pair.')

parser.add_argument('--load_vocab', action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--load_dataset', action='store_true', help='load a pre-processed dataset')
parser.add_argument('--use_revtok', action='store_true', help='use reversible tokenization')
parser.add_argument('--level', type=str, default='subword', help='for BPE, we must preprocess the dataset')
parser.add_argument('--good_course', action='store_true', help='use beam-search output for distillation')
parser.add_argument('--test_set', type=str, default=None, help='which test set to use')
parser.add_argument('--max_len', type=int, default=None, help='limit the train set sentences to this many tokens')

parser.add_argument('--remove_eos', action='store_true', help='possibly remove <eos> tokens for FastTransformer')

# model basic
parser.add_argument('--prefix', type=str, default='', help='prefix to denote the model, nothing or [time]')
parser.add_argument('--params', type=str, default='james-iwslt', help='pamarater sets: james-iwslt, t2t-base, etc')
parser.add_argument('--fast', dest='model', action='store_const', const=FastTransformer,
                    default=Transformer, help='use a single self-attn stack')

# model variants
parser.add_argument('--local', dest='windows', action='store_const', const=[1, 3, 5, 7, -1],
                    default=None, help='use local attention')
parser.add_argument('--causal', action='store_true', help='use causal attention')
parser.add_argument('--positional_attention', action='store_true', help='incorporate positional information in key/value')
parser.add_argument('--no_source', action='store_true')
parser.add_argument('--use_mask', action='store_true', help='use src/trg mask during attention')
parser.add_argument('--diag', action='store_true', help='ignore diagonal attention when doing self-attention.')
parser.add_argument('--convblock', action='store_true', help='use ConvBlock instead of ResNet')
parser.add_argument('--cosine_output', action='store_true', help='use cosine similarity as output layer')

parser.add_argument('--noisy', action='store_true', help='inject noise in the attention mechanism: Beta-Gumbel softmax')
parser.add_argument('--noise_samples', type=int, default=0, help='only useful for noisy parallel decoding')

parser.add_argument('--critic', action='store_true', help='use critic')
parser.add_argument('--kernel_sizes', type=str, default='2,3,4,5', help='kernel sizes of convnet critic')
parser.add_argument('--kernel_num', type=int, default=128, help='number of each kind of kernel')

parser.add_argument('--use_wo', action='store_true', help='use output weight matrix in multihead attention')
parser.add_argument('--share_embeddings', action='store_true', help='share embeddings between encoder and decoder')

parser.add_argument('--use_alignment', action='store_true', help='use the aligned fake data to initialize')
parser.add_argument('--hard_inputs', action='store_true', help='use hard selection as inputs, instead of soft-attention over embeddings.')
parser.add_argument('--preordering', action='store_true', help='use the ground-truth reordering information')
parser.add_argument('--use_posterior_order', action='store_true', help='directly use the groud-truth alignment for reordering.')
parser.add_argument('--train_decoder_with_order', action='store_true', help='when training the decoder, use the ground-truth')

parser.add_argument('--postordering', action='store_true', help='just have a try...')
parser.add_argument('--fertility_only', action='store_true')
parser.add_argument('--highway', action='store_true', help='usually false')
parser.add_argument('--mix_of_experts', action='store_true')
parser.add_argument('--orderless', action='store_true', help='for the inputs, remove the order information')
parser.add_argument('--cheating', action='store_true', help='disable decoding, always use real fertility')

# running
parser.add_argument('--mode', type=str, default='train', help='train, test or build')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use or -1 for CPU')
parser.add_argument('--seed', type=int, default=19920206, help='seed for randomness')

parser.add_argument('--eval-every', type=int, default=1000, help='run dev every')
parser.add_argument('--maximum_steps', type=int, default=1000000, help='maximum steps you take to train a model')
parser.add_argument('--disable_lr_schedule', action='store_true', help='disable the transformer learning rate')
parser.add_argument('--batchsize', type=int, default=2048, help='# of tokens processed per batch')

parser.add_argument('--hidden_size', type=int, default=None, help='input the hidden size')
parser.add_argument('--length_ratio', type=int, default=2, help='maximum lengths of decoding')
parser.add_argument('--optimizer', type=str, default='Adam')

parser.add_argument('--beam_size', type=int, default=1, help='beam-size used in Beamsearch, default using greedy decoding')
parser.add_argument('--alpha', type=float, default=0.6, help='length normalization weights')
parser.add_argument('--temperature', type=float, default=1, help='smoothing temperature for noisy decoding')
parser.add_argument('--multi_run', type=int, default=1, help='we can run the code multiple times to get the best')

parser.add_argument('--load_from', type=str, default=None, help='load from checkpoint')
parser.add_argument('--resume', action='store_true', help='when loading from the saved model, it resumes from that.')
parser.add_argument('--teacher', type=str, default=None, help='load a pre-trained auto-regressive model.')
parser.add_argument('--share_encoder', action='store_true', help='use teacher-encoder to initialize student')
parser.add_argument('--finetune_encoder', action='store_true', help='if further train the encoder')

parser.add_argument('--seq_dist',  action='store_true', help='knowledge distillation at sequence level')
parser.add_argument('--word_dist', action='store_true', help='knowledge distillation at word level')
parser.add_argument('--greedy_fertility', action='store_true', help='using the fertility generated by autoregressive model (only for seq_dist)')

parser.add_argument('--fertility_mode', type=str, default='argmax', help='mean, argmax or reinforce')
parser.add_argument('--finetuning_truth', action='store_true', help='use ground-truth for finetuning')

parser.add_argument('--trainable_teacher', action='store_true', help='have a trainable teacher')
parser.add_argument('--only_update_errors', action='store_true', help='have a trainable teacher')
parser.add_argument('--teacher_use_real', action='store_true', help='teacher also trained with MLE on real data')
parser.add_argument('--max_cache', type=int, default=0, help='save most recent max_cache decoded translations')
parser.add_argument('--replay_every', type=int, default=1000, help='every 1k updates, train the teacher again')
parser.add_argument('--replay_times', type=int, default=250, help='train the teacher again for 250k steps')

parser.add_argument('--margin', type=float, default=1.5, help='margin to make sure teacher will give higher score to real data')
parser.add_argument('--real_data', action='store_true', help='only used in the reverse kl setting')
parser.add_argument('--beta1', type=float, default=0.5, help='balancing MLE and KL loss.')
parser.add_argument('--beta2', type=float, default=0.01, help='balancing the GAN loss.')
parser.add_argument('--critic_only', type=int, default=0, help='pre-training the critic model.')
parser.add_argument('--st', action='store_true', help='straight through estimator')
parser.add_argument('--entropy', action='store_true')

parser.add_argument('--no_bpe', action='store_true', help='output files without BPE')
parser.add_argument('--no_write', action='store_true', help='do not write the decoding into the decoding files.')
parser.add_argument('--output_fer', action='store_true', help='decoding and output fertilities')

# debugging
parser.add_argument('--check', action='store_true', help='on training, only used to check on the test set.')
parser.add_argument('--debug', action='store_true', help='debug mode: no saving or tensorboard')
parser.add_argument('--tensorboard', action='store_true', help='use TensorBoard')

# old params
parser.add_argument('--old', action='store_true', help='this is used for solving conflicts of new codes')
parser.add_argument('--hyperopt', action='store_true', help='use HyperOpt')
parser.add_argument('--scst', action='store_true', help='use HyperOpt')

parser.add_argument('--serve', type=int, default=None, help='serve at port')
parser.add_argument('--attention_discrimination', action='store_true')

# ---------------------------------------------------------------------------------------------------------------- #

args = parser.parse_args()
if args.prefix == '[time]':
    args.prefix = strftime("%m.%d_%H.%M.", gmtime())
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

# get the langauage pairs:
args.src = args.language[:2]  # source language
args.trg = args.language[2:]  # target language

# logger settings
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler('./logs/log-{}.txt'.format(args.prefix))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# setup random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# setup data-field
DataField = data.ReversibleField if args.use_revtok else NormalField
tokenizer = revtok.tokenize if args.use_revtok else lambda x: x.replace('@@ ', '').split()

TRG   = DataField(init_token='<init>', eos_token='<eos>', batch_first=True)
SRC   = DataField(batch_first=True) if not args.share_embeddings else TRG
ALIGN = data.Field(sequential=True, preprocessing=data.Pipeline(lambda tok: int(tok.split('-')[0])), use_vocab=False, pad_token=0, batch_first=True)
FER   = data.Field(sequential=True, preprocessing=data.Pipeline(lambda tok: int(tok)), use_vocab=False, pad_token=0, batch_first=True)
align_dict, align_table = None, None

# setup many datasets (need to manaually setup)
data_prefix = args.data_prefix
if args.dataset == 'iwslt':
    if args.test_set is None:
        args.test_set = 'IWSLT16.TED.tst2013'
    if args.dist_set is None:
        args.dist_set = '.dec.b1'


    elif args.greedy_fertility:
        logger.info('use the fertility predicted by autoregressive model (instead of fast-align)')
        train_data, dev_data = ParallelDataset.splits(
        path=data_prefix + 'iwslt/en-de/', train='train.en-de.bpe.new',
        validation='IWSLT16.TED.tst2013.en-de.bpe.new.dev', exts=('.src.b1', '.trg.b1', '.dec.b1', '.fer', '.fer'),
        fields=[('src', SRC), ('trg', TRG), ('dec', TRG), ('fer', FER), ('fer_dec', FER)],
        load_dataset=args.load_dataset, prefix='ts')

    elif (args.mode == 'test') or (args.mode == 'test_noisy'):
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'iwslt/en-de/', train='train.tags.en-de{}'.format(
                '.bpe' if not args.use_revtok else ''),
            validation='{}.en-de{}'.format(
                args.test_set, '.bpe' if not args.use_revtok else ''), exts=('.en', '.de'),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='normal')

    else:
        train_data, dev_data = ParallelDataset.splits(
        path=data_prefix + 'iwslt/en-de/', train='train.tags.en-de.bpe',
        validation='train.tags.en-de.bpe.dev', exts=('.en2', '.de2', '.decoded2', '.aligned', '.decode.aligned', '.fer', '.decode.fer'),
        fields=[('src', SRC), ('trg', TRG), ('dec', TRG), ('align', ALIGN), ('align_dec', ALIGN), ('fer', FER), ('fer_dec', FER)],
        load_dataset=args.load_dataset, prefix='ts')

    decoding_path = data_prefix + 'iwslt/en-de/{}.en-de.bpe.new'
    if args.use_alignment and (args.model is FastTransformer):
        align_dict = {l.split()[0]: l.split()[1] for l in open(data_prefix + 'iwslt/en-de/train.tags.en-de.dict')}

elif args.dataset == 'wmt16-ende':
    if args.test_set is None:
        args.test_set = 'newstest2013'

    if (args.mode == 'test') or (args.mode == 'test_noisy'):
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'wmt16-ende/', train='newstest2013.tok.bpe.32000',
            validation='{}.tok.bpe.32000'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='real')
        decoding_path = data_prefix + 'wmt16-ende/test.{}.{}'.format(args.prefix, args.test_set)

    elif not args.seq_dist:
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'wmt16-ende/', train='train.tok.clean.bpe.32000',
            validation='{}.tok.bpe.32000'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='real')
        decoding_path = data_prefix + 'wmt16-ende/{}.tok.bpe.decode'
    else:
        train_data, dev_data = ParallelDataset.splits(
            path=data_prefix + 'wmt16-ende/', train='train.tok.bpe.decode',
            validation='newstest2013.tok.bpe.decode.dev',
            exts=('.src.b1', '.trg.b1', '.dec.b1', '.real.aligned', '.fake.aligned', '.real.fer', '.fake.fer'),
            fields=[('src', SRC), ('trg', TRG), ('dec', TRG), ('align', ALIGN), ('align_dec', ALIGN), ('fer', FER), ('fer_dec', FER)],
            load_dataset=args.load_dataset, prefix='ts')
        decoding_path = data_prefix + 'wmt16-ende/{}.tok.bpe.na'

    if args.use_alignment and (args.model is FastTransformer):
        align_table = {l.split()[0]: l.split()[1] for l in
                        open(data_prefix + 'wmt16-ende/train.tok.bpe.decode.full.fastlign2.dict')}

elif args.dataset == 'wmt16-deen':
    if args.test_set is None:
        args.test_set = 'newstest2013'

    if (args.mode == 'test') or (args.mode == 'test_noisy'):
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'wmt16-ende/', train='newstest2013.tok.bpe.32000',
            validation='{}.tok.bpe.32000'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='real')
        decoding_path = data_prefix + 'wmt16-ende/test.{}.{}'.format(args.prefix, args.test_set)

    elif not args.seq_dist:
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'wmt16-deen/', train='train.tok.clean.bpe.32000',
            validation='{}.tok.bpe.32000'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='real')
        decoding_path = data_prefix + 'wmt16-deen/{}.tok.bpe.decode'

    else:
        train_data, dev_data = ParallelDataset.splits(
            path=data_prefix + 'wmt16-deen/', train='train.tok.bpe.decode',
            validation='{}.tok.bpe.decode.dev'.format(args.test_set),
            exts=('.src.b1', '.trg.b1', '.dec.b1', '.real.aligned', '.fake.aligned', '.real.fer', '.fake.fer'),
            fields=[('src', SRC), ('trg', TRG), ('dec', TRG), ('align', ALIGN), ('align_dec', ALIGN), ('fer', FER), ('fer_dec', FER)],
            load_dataset=args.load_dataset, prefix='ts')
        decoding_path = data_prefix + 'wmt16-deen/{}.tok.bpe.na'

    if args.use_alignment and (args.model is FastTransformer):
        align_table = {l.split()[0]: l.split()[1] for l in
                        open(data_prefix + 'wmt16-deen/train.tok.bpe.decode.full.fastlign2.dict')}

elif args.dataset == 'wmt16-enro':
    if args.test_set is None:
        args.test_set = 'dev'

    if (args.mode == 'test') or (args.mode == 'test_noisy'):
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'wmt16-enro/', train='dev.bpe',
            validation='{}.bpe'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='real')
        decoding_path = data_prefix + 'wmt16-enro/{}.bpe.decode'

    elif not args.seq_dist:
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'wmt16-enro/', train='corpus.bpe',
            validation='{}.bpe'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='real')
        decoding_path = data_prefix + 'wmt16-enro/{}.bpe.decode'

    else:
        train_data, dev_data = ParallelDataset.splits(
            path=data_prefix + 'wmt16-enro/', train='train.bpe.decode',
            validation='dev.bpe.decode.dev',
            exts=('.src.b1', '.trg.b1', '.dec.b1', '.real.aligned', '.fake.aligned', '.real.fer', '.fake.fer'),
            fields=[('src', SRC), ('trg', TRG), ('dec', TRG), ('align', ALIGN), ('align_dec', ALIGN), ('fer', FER), ('fer_dec', FER)],
            load_dataset=args.load_dataset, prefix='ts')
        decoding_path = data_prefix + 'wmt16-enro/{}.tok.bpe.na'

    if args.use_alignment and (args.model is FastTransformer):
        align_table = {l.split()[0]: l.split()[1] for l in
                        open(data_prefix + 'wmt16-enro/train.bpe.decode.full.fastlign2.dict')}

elif args.dataset == 'wmt16-roen':
    if args.test_set is None:
        args.test_set = 'dev'

    if (args.mode == 'test') or (args.mode == 'test_noisy'):
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'wmt16-roen/', train='dev.bpe',
            validation='{}.bpe'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='real')
        decoding_path = data_prefix + 'wmt16-roen/{}.bpe.decode'

    elif not args.seq_dist:
        train_data, dev_data = NormalTranslationDataset.splits(
            path=data_prefix + 'wmt16-roen/', train='corpus.bpe',
            validation='{}.bpe'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
            fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='real')
        decoding_path = data_prefix + 'wmt16-roen/{}.bpe.decode'

    else:
        train_data, dev_data = ParallelDataset.splits(
            path=data_prefix + 'wmt16-roen/', train='train.bpe.decode',
            validation='dev.bpe.decode.dev',
            exts=('.src.b1', '.trg.b1', '.dec.b1', '.real.aligned', '.fake.aligned', '.real.fer', '.fake.fer'),
            fields=[('src', SRC), ('trg', TRG), ('dec', TRG), ('align', ALIGN), ('align_dec', ALIGN), ('fer', FER), ('fer_dec', FER)],
            load_dataset=args.load_dataset, prefix='ts')
        decoding_path = data_prefix + 'wmt16-roen/{}.tok.bpe.na'

    if args.use_alignment and (args.model is FastTransformer):
        align_table = {l.split()[0]: l.split()[1] for l in
                        open(data_prefix + 'wmt16-roen/train.bpe.decode.full.fastlign2.dict')}

else:
    raise NotImplementedError


# build word-level vocabularies
if args.load_vocab and os.path.exists(data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg))):

    logger.info('load saved vocabulary.')
    src_vocab, trg_vocab = torch.load(data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg)))
    SRC.vocab = src_vocab
    TRG.vocab = trg_vocab

else:

    logger.info('save the vocabulary')
    if not args.share_embeddings:
        SRC.build_vocab(train_data, dev_data, max_size=50000)
    TRG.build_vocab(train_data, dev_data, max_size=50000)
    torch.save([SRC.vocab, TRG.vocab], data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg)))
args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

# build alignments ---
if align_dict is not None:
    align_table = [TRG.vocab.stoi['<init>'] for _ in range(len(SRC.vocab.itos))]
    for src in align_dict:
        align_table[SRC.vocab.stoi[src]] = TRG.vocab.stoi[align_dict[src]]
    align_table[0] = 0  # --<unk>
    align_table[1] = 1  # --<pad>

def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    if args.seq_dist:
        return max(len(new.src), len(new.trg), len(new.dec), prev_max_len) * i
    else:
        return max(len(new.src), len(new.trg),  prev_max_len) * i

def dyn_batch_without_padding(new, i, sofar):
    if args.seq_dist:
        return sofar + max(len(new.src), len(new.trg), len(new.dec))
    else:
        return sofar + max(len(new.src), len(new.trg))
# build the dataset iterators

# work around torchtext making it hard to share vocabs without sharing other field properties
if args.share_embeddings:
    SRC = copy.deepcopy(SRC)
    SRC.init_token = None
    SRC.eos_token = None
    train_data.fields['src'] = SRC
    dev_data.fields['src'] = SRC

if (args.model is FastTransformer) and (args.remove_eos):
    TRG.eos_token = None

if args.max_len is not None:
    train_data.examples = [ex for ex in train_data.examples if len(ex.trg) <= args.max_len]

if args.batchsize == 1:  # speed-test: one sentence per batch.
    batch_size_fn = lambda new, count, sofar: count
else:
    batch_size_fn = dyn_batch_without_padding if args.model is Transformer else dyn_batch_with_padding

train_real, dev_real = data.BucketIterator.splits(
    (train_data, dev_data), batch_sizes=(args.batchsize, args.batchsize), device=args.gpu,
    batch_size_fn=batch_size_fn,
    repeat=None if args.mode == 'train' else False)

logger.info("build the dataset. done!")

# model hyper-params:
hparams = None
if args.dataset == 'iwslt':
    if args.params == 'james-iwslt':
        hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746} # ~32
    elif args.params == 'james-iwslt2':
        hparams = {'d_model': 278, 'd_hidden': 2048, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746} # ~32
    teacher_hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746}

elif args.dataset == 'wmt16-ende':
    logger.info('use default parameters of t2t-base')
    hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
    teacher_hparams = hparams

elif args.dataset == 'wmt16-deen':
    logger.info('use default parameters of t2t-base')
    hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
    teacher_hparams = hparams

elif args.dataset == 'wmt16-enro':
    logger.info('use default parameters of t2t-base')
    hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
    teacher_hparams = hparams

elif args.dataset == 'wmt16-roen':
    logger.info('use default parameters of t2t-base')
    hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
    teacher_hparams = hparams

if hparams is None:
    logger.info('use default parameters of t2t-base')
    hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32

if args.teacher is not None:
    teacher_args = copy.deepcopy(args)
    teacher_args.__dict__.update(teacher_hparams)
args.__dict__.update(hparams)
if args.hidden_size is not None:
    args.d_hidden = args.hidden_size

# show the arg:
logger.info(args)

hp_str = (f"{args.dataset}_{args.level}_{'fast_' if args.model is FastTransformer else ''}"
        f"{args.d_model}_{args.d_hidden}_{args.n_layers}_{args.n_heads}_"
        f"{args.drop_ratio:.3f}_{args.warmup}_"
        f"{args.xe_until if hasattr(args, 'xe_until') else ''}_"
        f"{f'{args.xe_ratio:.3f}' if hasattr(args, 'xe_ratio') else ''}_"
        f"{args.xe_every if hasattr(args, 'xe_every') else ''}")
logger.info(f'Starting with HPARAMS: {hp_str}')

model_name = './models/' + args.prefix + hp_str

# build the model
model = args.model(SRC, TRG, args)
if args.load_from is not None:
    with torch.cuda.device(args.gpu):   # very important.
        model.load_state_dict(torch.load('./models/' + args.load_from + '.pt',
        map_location=lambda storage, loc: storage.cuda()))  # load the pretrained models.
if args.critic:
    model.install_critic()

# logger.info(str(model))

# if using a teacher
if args.teacher is not None:
    teacher_model = Transformer(SRC, TRG, teacher_args)
    with torch.cuda.device(args.gpu):
        teacher_model.load_state_dict(torch.load('./models/' + args.teacher + '.pt',
                                    map_location=lambda storage, loc: storage.cuda()))
    for params in teacher_model.parameters():
        if args.trainable_teacher:
            params.requires_grad = True
        else:
            params.requires_grad = False

    if (args.share_encoder) and (args.load_from is None):
        model.encoder = copy.deepcopy(teacher_model.encoder)
        for params in model.encoder.parameters():
            if args.finetune_encoder:
                params.requires_grad = True
            else:
                params.requires_grad = False

else:
    teacher_model = None

# use cuda
if args.gpu > -1:
    model.cuda(args.gpu)
    if align_table is not None:
        align_table = torch.LongTensor(align_table).cuda(args.gpu)
        align_table = Variable(align_table)
        model.alignment = align_table

    if args.teacher is not None:
        teacher_model.cuda(args.gpu)

def register_nan_checks(m):
    def check_grad(module, grad_input, grad_output):
        if any(np.any(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
            print('NaN gradient in ' + type(module).__name__)
            1/0
    m.apply(lambda module: module.register_backward_hook(check_grad))

def get_learning_rate(i, lr0=0.1):
        if not args.disable_lr_schedule:
            return lr0 * 10 / math.sqrt(args.d_model) * min(
                1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
        return 0.00002

def export(x):
    try:
        with torch.cuda.device(args.gpu):
            return x.data.cpu().float().mean()
    except Exception:
        return 0

def devol(batch):
    new_batch = copy.copy(batch)
    new_batch.src = Variable(batch.src.data, volatile=True)
    return new_batch

# register_nan_checks(model)
# register_nan_checks(teacher_model)

def valid_model(model, dev, dev_metrics=None, distillation=False, print_out=False, teacher_model=None):
    print_seqs = ['[sources]', '[targets]', '[decoded]', '[fertili]', '[origind]']
    trg_outputs, dec_outputs = [], []
    outputs = {}

    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    for j, dev_batch in enumerate(dev):

        # decode from the model (whatever Transformer or FastTransformer)
        torch.cuda.nvtx.range_push('quick_prepare')
        inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(dev_batch, distillation)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push('prepare_initial')
        decoder_inputs, input_reorder, reordering_cost = inputs, None, None
        if type(model) is FastTransformer:
            # batch_align = dev_batch.align_dec if distillation else dev_batch.align
            batch_align = None
            batch_fer   = dev_batch.fer_dec if distillation else dev_batch.fer

            # if args.postordering:
            #
            #     targets_sorted = targets.gather(1, align_index)
            # batch_align_sorted, align_index = masked_sort(batch_align, target_masks)  # change the target indexxx, batch x max_trg
            decoder_inputs, input_reorder, decoder_masks, reordering_cost = model.prepare_initial(encoding,
                                    sources, source_masks, input_masks,
                                    batch_align, batch_fer, decoding=(not args.cheating), mode='argmax')
        else:
            decoder_masks = input_masks

        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push('model')
        decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks, decoding=True, return_probs=True)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push('batched_cost')
        loss = 0

        if args.postordering:
            if args.cheating:
                decoding1 = unsorted(decoding, align_index)
            else:
                positions = model.predict_offset(out, decoder_masks, None)
                shifted_index = positions.sort(1)[1]
                decoding1 = unsorted(decoding, shifted_index)
        else:
            decoding1 = decoding

        # loss = model.batched_cost(targets, target_masks, probs)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push('output_decoding')
        dev_outputs = [model.output_decoding(d) for d in [('src', sources), ('trg', targets), ('trg', decoding1), ('src', input_reorder)]]
        if args.postordering:
            dev_outputs += [model.output_decoding(('trg', decoding))]

        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push('computeGLEU')
        gleu = computeGLEU(dev_outputs[2], dev_outputs[1], corpus=False, tokenizer=tokenizer)
        torch.cuda.nvtx.range_pop()

        if print_out:
            for k, d in enumerate(dev_outputs):
                logger.info("{}: {}".format(print_seqs[k], d[0]))
            logger.info('------------------------------------------------------------------')

        if teacher_model is not None:  # teacher is Transformer, student is FastTransformer
            inputs_student, _, targets_student, _, _, _, encoding_teacher, _ = teacher_model.quick_prepare(dev_batch, False, decoding, decoding,
                                                                                                        input_masks, target_masks, source_masks)
            teacher_real_loss  = teacher_model.cost(targets, target_masks,
                                out=teacher_model(encoding_teacher, source_masks, inputs, input_masks))

            teacher_fake_out   = teacher_model(encoding_teacher, source_masks, inputs_student, input_masks)
            teacher_fake_loss  = teacher_model.cost(targets_student, target_masks, out=teacher_fake_out)
            teacher_alter_loss = teacher_model.cost(targets, target_masks, out=teacher_fake_out)

        trg_outputs += dev_outputs[1]
        dec_outputs += dev_outputs[2]

        if dev_metrics is not None:

            values = [loss, gleu]
            if teacher_model is not None:
                values  += [teacher_real_loss, teacher_fake_loss,
                            teacher_real_loss - teacher_fake_loss,
                            teacher_alter_loss,
                            teacher_alter_loss - teacher_fake_loss]
            if reordering_cost is not None:
                values += [reordering_cost]

            dev_metrics.accumulate(batch_size, *values)

    corpus_gleu = computeGLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
    corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
    outputs['corpus_gleu'] = corpus_gleu
    outputs['corpus_bleu'] = corpus_bleu
    if dev_metrics is not None:
        logger.info(dev_metrics)
    logger.info("The dev-set corpus GLEU = {}".format(corpus_gleu))
    logger.info("The dev-set corpus BLEU = {}".format(corpus_bleu))
    return outputs


def train_model(model, train, dev, teacher_model=None):

    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('./runs/{}'.format(args.prefix+hp_str))

    # optimizer
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
        if args.trainable_teacher:
            opt_teacher = torch.optim.Adam([p for p in teacher_model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
    elif args.optimizer == 'RMSprop':
        opt = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], eps=1e-9)
        if args.trainable_teacher:
            opt_teacher = torch.optim.RMSprop([p for p in teacher_model.parameters() if p.requires_grad], eps=1e-9)
    else:
        raise NotImplementedError

    # if resume training
    if (args.load_from is not None) and (args.resume):
        with torch.cuda.device(args.gpu):   # very important.
            offset, opt_states = torch.load('./models/' + args.load_from + '.pt.states',
                                            map_location=lambda storage, loc: storage.cuda())
            opt.load_state_dict(opt_states)
    else:
        offset = 0

    # metrics
    best = Best(max, 'corpus_bleu', 'corpus_gleu', 'gleu', 'loss', 'i', model=model, opt=opt, path=model_name, gpu=args.gpu)
    train_metrics = Metrics('train', 'loss', 'real', 'fake')
    dev_metrics = Metrics('dev', 'loss', 'gleu', 'real_loss', 'fake_loss', 'distance', 'alter_loss', 'distance2', 'reordering_loss', 'corpus_gleu')
    progressbar = tqdm(total=args.eval_every, desc='start training.')

    # cache
    if args.max_cache > 0:
        caches = Cache(args.max_cache, args.gpu)

    for iters, batch in enumerate(train):
        iters += offset
        if iters > args.maximum_steps:
            logger.info('reach the maximum updating steps.')
            break

        if iters % args.eval_every == 0:
            progressbar.close()
            dev_metrics.reset()

            if args.seq_dist:
                outputs_course = valid_model(model, dev, dev_metrics,
                        distillation=True, teacher_model=None)#teacher_model=teacher_model)

            if args.trainable_teacher:
                outputs_teacher = valid_model(teacher_model, dev, None)

            outputs_data = valid_model(model, dev, None if args.seq_dist else dev_metrics, teacher_model=None, print_out=True)

            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/GLEU_sentence_', dev_metrics.gleu, iters)
                writer.add_scalar('dev/Loss', dev_metrics.loss, iters)
                writer.add_scalar('dev/GLEU_corpus_', outputs_data['corpus_gleu'], iters)
                writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], iters)

                if args.seq_dist:
                    writer.add_scalar('dev/GLEU_corpus_dis', outputs_course['corpus_gleu'], iters)
                    writer.add_scalar('dev/BLEU_corpus_dis', outputs_course['corpus_bleu'], iters)

                if args.trainable_teacher:
                    writer.add_scalar('dev/GLEU_corpus_teacher', outputs_teacher['corpus_gleu'], iters)
                    writer.add_scalar('dev/BLEU_corpus_teacher', outputs_teacher['corpus_bleu'], iters)

                if args.teacher is not None:
                    writer.add_scalar('dev/Teacher_real_loss', dev_metrics.real_loss, iters)
                    writer.add_scalar('dev/Teacher_fake_loss', dev_metrics.fake_loss, iters)
                    writer.add_scalar('dev/Teacher_alter_loss', dev_metrics.alter_loss, iters)
                    writer.add_scalar('dev/Teacher_distance',  dev_metrics.distance, iters)
                    writer.add_scalar('dev/Teacher_distance2', dev_metrics.distance2, iters)

                if args.preordering:
                    writer.add_scalar('dev/Reordering_loss', dev_metrics.reordering_loss, iters)

            if not args.debug:
                best.accumulate(outputs_data['corpus_bleu'], outputs_data['corpus_gleu'], dev_metrics.gleu, dev_metrics.loss, iters)
                logger.info('the best model is achieved at {}, average greedy GLEU={}, corpus GLEU={}, corpus BLEU={}'.format(
                    best.i, best.gleu, best.corpus_gleu, best.corpus_bleu))
            logger.info('model:' + args.prefix + hp_str)

            # ---set-up a new progressor---
            progressbar = tqdm(total=args.eval_every, desc='start training.')

        # --- training --- #
        # try:
        model.train()
        opt.param_groups[0]['lr'] = get_learning_rate(iters + 1)
        opt.zero_grad()

        # prepare the data
        inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(batch, args.seq_dist)
        input_reorder, reordering_cost, decoder_inputs = None, None, inputs
        batch_align = None # batch.align_dec if args.seq_dist else batch.align
        batch_fer   = batch.fer_dec   if args.seq_dist  else batch.fer
        # batch_align_sorted, align_index = masked_sort(batch_align, target_masks)  # change the target indexxx, batch x max_trg

        # print(batch_fer.size(), input_masks.size(), source_masks.size(), sources.size())

        # Prepare_Initial
        if type(model) is FastTransformer:
            inputs, input_reorder, input_masks, reordering_cost = model.prepare_initial(encoding, sources, source_masks, input_masks, batch_align, batch_fer)

        # Maximum Likelihood Training
        feedback = {}
        if not args.word_dist:
            loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks, positions= None, feedback=feedback))

            # train the reordering also using MLE??
            if args.preordering:
                loss += reordering_cost

        else:
            # only used for FastTransformer: word-level adjustment

            if not args.preordering:
                decoding, out, probs = model(encoding, source_masks, inputs, input_masks, return_probs=True, decoding=True)
                loss_student = model.batched_cost(targets, target_masks, probs)  # student-loss (MLE)
                decoder_masks = input_masks

            else: # Note that MLE and decoding has different translations. We need to run the same code twice

                if args.finetuning_truth:
                    decoding, out, probs = model(encoding, source_masks, inputs, input_masks, decoding=True, return_probs=True, feedback=feedback)
                    loss_student = model.cost(targets, target_masks, out=out)
                    decoder_masks = input_masks

                else:
                    if args.fertility_mode != 'reinforce':
                        loss_student = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks, positions=None, feedback=feedback))
                        decoder_inputs, _, decoder_masks, _ = model.prepare_initial(encoding, sources, source_masks, input_masks,
                                                                                    batch_align, batch_fer, decoding=True, mode=args.fertility_mode)
                        decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks, decoding=True, return_probs=True)  # decode again
                    else:
                        # truth
                        decoding, out, probs = model(encoding, source_masks, inputs, input_masks, decoding=True, return_probs=True, feedback=feedback)
                        loss_student = model.cost(targets, target_masks, out=out)
                        decoder_masks = input_masks

                        # baseline
                        decoder_inputs_b, _, decoder_masks_b, _ = model.prepare_initial(encoding, sources, source_masks, input_masks,
                                                                                        batch_align, batch_fer, decoding=True, mode='mean')
                        decoding_b, out_b, probs_b = model(encoding, source_masks, decoder_inputs_b, decoder_masks_b, decoding=True, return_probs=True)  # decode again

                        # reinforce
                        decoder_inputs_r, _, decoder_masks_r, _ = model.prepare_initial(encoding, sources, source_masks, input_masks,
                                                                                        batch_align, batch_fer, decoding=True, mode='reinforce')
                        decoding_r, out_r, probs_r = model(encoding, source_masks, decoder_inputs_r, decoder_masks_r, decoding=True, return_probs=True)  # decode again

            # train the reordering also using MLE??
            if args.preordering:
                loss_student += reordering_cost

            # teacher tries translation + look-at student's output
            teacher_model.eval()
            if args.fertility_mode != 'reinforce':
                inputs_student_index, _, targets_student_soft, _, _, _, encoding_teacher, _ = model.quick_prepare(batch, False, decoding, probs, decoder_masks, decoder_masks, source_masks)
                out_teacher, probs_teacher = teacher_model(encoding_teacher, source_masks, inputs_student_index.detach(), decoder_masks, return_probs=True)
                loss_teacher = teacher_model.batched_cost(targets_student_soft, decoder_masks, probs_teacher.detach())
                loss = (1 - args.beta1) * loss_teacher + args.beta1 * loss_student   # final results

            else:
                inputs_student_index, _, targets_student_soft, _, _, _, encoding_teacher, _ = model.quick_prepare(batch, False, decoding, probs, decoder_masks, decoder_masks, source_masks)
                out_teacher, probs_teacher = teacher_model(encoding_teacher, source_masks, inputs_student_index.detach(), decoder_masks, return_probs=True)
                loss_teacher = teacher_model.batched_cost(targets_student_soft, decoder_masks, probs_teacher.detach())

                inputs_student_index, _ = model.prepare_inputs(batch, decoding_b, False, decoder_masks_b)
                targets_student_soft, _ = model.prepare_targets(batch, probs_b, False, decoder_masks_b)

                out_teacher, probs_teacher = teacher_model(encoding_teacher, source_masks, inputs_student_index.detach(), decoder_masks_b, return_probs=True)

                _, loss_1= teacher_model.batched_cost(targets_student_soft, decoder_masks_b, probs_teacher.detach(), True)

                inputs_student_index, _ = model.prepare_inputs(batch, decoding_r, False, decoder_masks_r)
                targets_student_soft, _ = model.prepare_targets(batch, probs_r, False, decoder_masks_r)

                out_teacher, probs_teacher = teacher_model(encoding_teacher, source_masks, inputs_student_index.detach(), decoder_masks_r, return_probs=True)
                _, loss_2= teacher_model.batched_cost(targets_student_soft, decoder_masks_r, probs_teacher.detach(), True)

                rewards = -(loss_2 - loss_1).data
                # if rewards.size(0) != 1:
                rewards = rewards - rewards.mean()  # ) / (rewards.std() + TINY)
                rewards = rewards.expand_as(source_masks)
                rewards = rewards * source_masks

                # print(model.predictor.saved_fertilities)
                # print(batch.src.size())
                model.predictor.saved_fertilities.reinforce(0.1 * rewards.contiguous().view(-1, 1))
                loss = (1 - args.beta1) * loss_teacher + args.beta1 * loss_student #+ 0 * model.predictor.saved_fertilities.float().sum()   # detect reinforce
                # loss = 0 * model.predictor.saved_fertilities.float().sum()   # detect reinforce

        # accmulate the training metrics
        train_metrics.accumulate(batch_size, loss, print_iter=None)
        train_metrics.reset()

        # train the student
        if args.preordering and args.fertility_mode == 'reinforce':
            torch.autograd.backward((loss, model.predictor.saved_fertilities),
                                    (torch.ones(1).cuda(loss.get_device()), None))
        else:
            loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1)

        opt.step()

        info = 'training step={}, loss={:.3f}, lr={:.5f}'.format(iters, export(loss), opt.param_groups[0]['lr'])
        if args.word_dist:
            info += '| NA:{:.3f}, AR:{:.3f}'.format(export(loss_student), export(loss_teacher))

        if args.trainable_teacher and (args.max_cache <= 0):
            loss_alter, loss_worse = export(loss_alter), export(loss_worse)
            info += '| AL:{:.3f}, WO:{:.3f}'.format(loss_alter, loss_worse)

        if args.preordering:
            info += '| RE:{:.3f}'.format(export(reordering_cost))

        if args.fertility_mode == 'reinforce':
            info += '| RL: {:.3f}'.format(export(rewards.mean()))

        if args.max_cache > 0:
            info += '| caches={}'.format(len(caches.cache))

        if args.tensorboard and (not args.debug):
            writer.add_scalar('train/Loss', export(loss), iters)

        progressbar.update(1)
        progressbar.set_description(info)

        # continue-training the teacher model
        if args.trainable_teacher:
            if args.max_cache > 0:
                caches.add([batch.src, batch.trg, batch.dec, decoding]) # experience-reply

            # trainable teacher: used old experience to train
            if (iters+1) % args.replay_every == 0:
                # ---set-up a new progressor: teacher training--- #
                progressbar_teacher = tqdm(total=args.replay_times, desc='start training the teacher.')

                for j in range(args.replay_times):

                    opt_teacher.param_groups[0]['lr'] = get_learning_rate(iters + 1)
                    opt_teacher.zero_grad()

                    src, trg, dec, decoding = caches.sample()
                    batch = Batch(src, trg, dec)

                    inputs, input_masks, targets, target_masks, sources, source_masks, encoding_teacher, batch_size = teacher_model.quick_prepare(batch, (not args.teacher_use_real))
                    inputs_students, _ = teacher_model.prepare_inputs(batch, decoding, masks=input_masks)
                    loss_alter = teacher_model.cost(targets, target_masks, out=teacher_model(encoding_teacher, source_masks, inputs_students, input_masks))
                    loss_worse = teacher_model.cost(targets, target_masks, out=teacher_model(encoding_teacher, source_masks, inputs, input_masks))

                    loss2  = loss_alter + loss_worse
                    loss2.backward()
                    opt_teacher.step()

                    info = 'teacher step={}, loss={:.3f}, alter={:.3f}, worse={:.3f}'.format(j, export(loss2), export(loss_alter), export(loss_worse))
                    progressbar_teacher.update(1)
                    progressbar_teacher.set_description(info)
                progressbar_teacher.close()
        # except Exception as e:
        #     logger.warn('caught an exception: {}'.format(e))


def decode_model(model, train_real, dev_real, evaluate=True, decoding_path=None, names=['en', 'de', 'decode']):

    if train_real is None:
        logger.info('decoding from the devlopment set. beamsize={}, alpha={}'.format(args.beam_size, args.alpha))
        dev = dev_real
    else:
        logger.info('decoding from the training set. beamsize={}, alpha={}'.format(args.beam_size, args.alpha))
        dev = train_real
        dev.train = False # make the Iterator create Variables with volatile=True so no graph is built

    progressbar = tqdm(total=sum([1 for _ in dev]), desc='start decoding')
    model.eval()

    if decoding_path is not None:
        decoding_path = decoding_path.format(args.test_set if train_real is None else 'train')
        handle_dec = open(decoding_path + '.{}'.format(names[2]), 'w')
        handle_src = open(decoding_path + '.{}'.format(names[0]), 'w')
        handle_trg = open(decoding_path + '.{}'.format(names[1]), 'w')
        if args.output_fer:
            handle_fer = open(decoding_path + '.{}'.format('fer'), 'w')

    corpus_size = 0
    src_outputs, trg_outputs, dec_outputs, timings = [], [], [], []
    decoded_words, target_words, decoded_info = 0, 0, 0

    attentions = None #{'source': None, 'target': None}
    pad_id = model.decoder.field.vocab.stoi['<pad>']
    eos_id = model.decoder.field.vocab.stoi['<eos>']

    curr_time = 0
    for iters, dev_batch in enumerate(dev):

        start_t = time.time()

        inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(dev_batch)

        if args.model is FastTransformer:
            decoder_inputs, input_reorder, decoder_masks, _ = model.prepare_initial(encoding, sources, source_masks, input_masks,
                                                                                None, None, decoding=True, mode=args.fertility_mode)
        else:
            decoder_inputs, decoder_masks = inputs, input_masks

        decoding = model(encoding, source_masks, decoder_inputs, decoder_masks, beam=args.beam_size, alpha=args.alpha, decoding=True, feedback=attentions)

        used_t = time.time() - start_t
        curr_time += used_t

        real_mask = 1 - ((decoding.data == eos_id) + (decoding.data == pad_id)).float()
        outputs = [model.output_decoding(d) for d in [('src', sources), ('trg', targets), ('trg', decoding)]]

        def DHondt(approx, mask):
            L = mask.size(1)
            w = torch.arange(1, 2 * L, 2)
            if approx.is_cuda:
                w = w.cuda(approx.get_device())
            w = 1 / w  # 1, 1/2, 1/3, ...
            approx = approx[:, :, None] @ w[None, :]  # B x Ts x Tt
            approx = approx.view(approx.size(0), -1)  # B x (Ts x Tt)
            appinx = approx.topk(L, 1)[1]             # B x Tt (index)

            fertility = approx.new(*approx.size()).fill_(0).scatter_(1, appinx, mask)
            fertility = fertility.contiguous().view(mask.size(0), -1, mask.size(1)).sum(2).long()
            return fertility

        def cutoff(s, t):
            for i in range(len(s), 0, -1):
                if s[i-1] != t:
                    return s[:i]
            raise IndexError

        if args.output_fer:
            source_attention = attentions['source'].data.mean(1).transpose(2, 1)  # B x Ts x Tt
            source_attention *= real_mask[:, None, :]
            approx_fertility = source_attention.sum(2)   # B x Ts
            fertility = DHondt(approx_fertility, real_mask)

        corpus_size += batch_size
        src_outputs += outputs[0]
        trg_outputs += outputs[1]
        dec_outputs += outputs[2]
        timings += [used_t]

        if decoding_path is not None:
            for s, t, d in zip(outputs[0], outputs[1], outputs[2]):
                if args.no_bpe:
                    s, t, d = s.replace('@@ ', ''), t.replace('@@ ', ''), d.replace('@@ ', '')

                print(s, file=handle_src, flush=True)
                print(t, file=handle_trg, flush=True)
                print(d, file=handle_dec, flush=True)


            if args.output_fer:
                with torch.cuda.device_of(fertility):
                    fertility = fertility.tolist()
                    for f in fertility:
                        f = ' '.join([str(fi) for fi in cutoff(f, 0)])
                        print(f, file=handle_fer, flush=True)

        progressbar.update(1)
        progressbar.set_description('finishing sentences={}/batches={}, speed={} sec/batch'.format(corpus_size, iters, curr_time / (1 + iters)))

    if evaluate:
        corpus_gleu = computeGLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
        corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
        logger.info("The dev-set corpus GLEU = {}".format(corpus_gleu))
        logger.info("The dev-set corpus BLEU = {}".format(corpus_bleu))

        computeGroupBLEU(dec_outputs, trg_outputs, tokenizer=tokenizer)
        torch.save([src_outputs, trg_outputs, dec_outputs, timings], './space/data.pt')


def noisy_decode_model(model, dev_real, samples=1, alpha=1, tau=1, teacher_model=None, evaluate=True,
                        decoding_path=None, names=['en', 'de', 'decode'], saveall=False):

    assert type(model) is FastTransformer, 'only works for fastTransformer'
    logger.info('decoding from the devlopment set. beamsize={}, alpha={}, tau={}'.format(args.beam_size, args.alpha, args.temperature))
    dev = dev_real

    progressbar = tqdm(total=sum([1 for _ in dev]), desc='start decoding')
    model.eval()
    teacher_model.eval()

    if decoding_path is not None:
        decoding_path = decoding_path.format(args.test_set if train_real is None else 'train')
        handle_dec = open(decoding_path + '.{}'.format(names[2]), 'w')
        handle_src = open(decoding_path + '.{}'.format(names[0]), 'w')
        handle_trg = open(decoding_path + '.{}'.format(names[1]), 'w')

        # if saveall:
        #     handle_fer = open(decoding_path + '.{}'.format(names[3]), 'w')

    corpus_size = 0
    src_outputs, trg_outputs, dec_outputs, timings = [], [], [], []
    all_dec_outputs = []

    decoded_words, target_words, decoded_info = 0, 0, 0

    attentions = None #{'source': None, 'target': None}
    pad_id = model.decoder.field.vocab.stoi['<pad>']
    eos_id = model.decoder.field.vocab.stoi['<eos>']

    curr_time = 0
    for iters, dev_batch in enumerate(dev):
        start_t = time.time()

        inputs, input_masks, targets, target_masks, sources, source_masks0, encoding0, batch_size = model.quick_prepare(dev_batch)
        if teacher_model is not None:
            encoding_teacher = teacher_model.encoding(sources, source_masks0)

        batch_size, src_len, hsize = encoding0[0].size()
        if samples > 1:
            source_masks = source_masks0[:, None, :].expand(batch_size, samples,
                src_len).contiguous().view(batch_size * samples, src_len)

            encoding = [None for _ in encoding0]
            for i in range(len(encoding)):
                encoding[i] = encoding0[i][:, None, :].expand(
                batch_size, samples, src_len, hsize).contiguous().view(batch_size * samples, src_len, hsize)

            if teacher_model is not None:
                for i in range(len(encoding)):
                    encoding_teacher[i] = encoding_teacher[i][:, None, :].expand(
                batch_size, samples, src_len, hsize).contiguous().view(batch_size * samples, src_len, hsize)

        def parallel():
            decoder_inputs, input_reorder, decoder_masks, logits_fer = model.prepare_initial(encoding0, sources, source_masks0, input_masks,
                                                                                            None, None, decoding=True, mode=args.fertility_mode, N=samples, tau=tau)
            if teacher_model is not None:
                decoding = model(encoding, source_masks, decoder_inputs, decoder_masks, decoding=True, feedback=attentions)
                student_inputs,  _ = teacher_model.prepare_inputs(dev_batch, decoding, decoder_masks)
                student_targets, _ = teacher_model.prepare_targets(dev_batch, decoding, decoder_masks)
                out, probs = teacher_model(encoding_teacher, source_masks, student_inputs, decoder_masks, return_probs=True, decoding=False)
                _, teacher_loss = model.batched_cost(student_targets, decoder_masks, probs, batched=True)  # student-loss (MLE)

                # reranking the translation
                teacher_loss = teacher_loss.view(batch_size, samples)
                decoding = decoding.view(batch_size, samples, -1)
                lp = decoder_masks.sum(1).view(batch_size, samples) ** (1 - alpha)
                teacher_loss = teacher_loss * Variable(lp)
            return decoding, teacher_loss, input_reorder

        if args.multi_run > 1:
            decodings, teacher_losses, _ = zip(*[parallel() for _ in range(args.multi_run)])
            maxl = max([d.size(2) for d in decodings])
            decoding = Variable(sources.data.new(batch_size, samples * args.multi_run, maxl).fill_(1).long())
            for i, d in enumerate(decodings):
                decoding[:, i * samples: (i+1) * samples, :d.size(2)] = d
            teacher_loss = torch.cat(teacher_losses, 1)
        else:
            decoding, teacher_loss, input_reorder = parallel()

        all_dec_outputs += [(decoding.view(batch_size * samples, -1), input_reorder)]

        selected_idx = (-teacher_loss).topk(1, 1)[1]   # batch x 1
        decoding = decoding.gather(1, selected_idx[:, :, None].expand(batch_size, 1, decoding.size(-1)))[:, 0, :]

        used_t = time.time() - start_t
        curr_time += used_t

        real_mask = 1 - ((decoding.data == eos_id) + (decoding.data == pad_id)).float()
        outputs = [model.output_decoding(d) for d in [('src', sources), ('trg', targets), ('trg', decoding)]]

        corpus_size += batch_size
        src_outputs += outputs[0]
        trg_outputs += outputs[1]
        dec_outputs += outputs[2]
        timings += [used_t]

        if decoding_path is not None:
            for s, t, d in zip(outputs[0], outputs[1], outputs[2]):
                if args.no_bpe:
                    s, t, d = s.replace('@@ ', ''), t.replace('@@ ', ''), d.replace('@@ ', '')
                print(s, file=handle_src, flush=True)
                print(t, file=handle_trg, flush=True)
                print(d, file=handle_dec, flush=True)

            # if saveall:
            #     for d, f in all_dec_outputs:
            #         ds = model.output_decoding(('trg', d))
            #         fs = model.output_decoding(('src', f))
            #         for dd, ff in zip(ds, fs):
            #             print(dd, file=handle_fer, flush=True)
            #             print(ff, file=handle_fer, flush=True)


        progressbar.update(1)
        progressbar.set_description('finishing sentences={}/batches={} speed={} sec/batch'.format(corpus_size, iters, curr_time / (1 + iters)))

    if evaluate:
        corpus_gleu = computeGLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
        corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
        logger.info("The dev-set corpus GLEU = {}".format(corpus_gleu))
        logger.info("The dev-set corpus BLEU = {}".format(corpus_bleu))

        computeGroupBLEU(dec_outputs, trg_outputs, tokenizer=tokenizer)
        torch.save([src_outputs, trg_outputs, dec_outputs, timings], './space/data.pt')


def self_improving_model(model, train, dev):
    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('./runs/self-{}'.format(args.prefix+hp_str))

    # optimizer
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
        if args.trainable_teacher:
            opt_teacher = torch.optim.Adam([p for p in teacher_model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
    elif args.optimizer == 'RMSprop':
        opt = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], eps=1e-9)
        if args.trainable_teacher:
            opt_teacher = torch.optim.RMSprop([p for p in teacher_model.parameters() if p.requires_grad], eps=1e-9)
    else:
        raise NotImplementedError

    # if resume training --
    if (args.load_from is not None) and (args.resume):
        with torch.cuda.device(args.gpu):   # very important.
            offset, opt_states = torch.load('./models/' + args.load_from + '.pt.states',
                                            map_location=lambda storage, loc: storage.cuda())
            opt.load_state_dict(opt_states)
    else:
        offset = 0

    # metrics
    best = Best(max, 'corpus_bleu', 'corpus_gleu', 'gleu', 'loss', 'i', model=model, opt=opt, path=model_name, gpu=args.gpu)
    train_metrics = Metrics('train', 'loss', 'real', 'fake')
    dev_metrics = Metrics('dev', 'loss', 'gleu', 'real_loss', 'fake_loss', 'distance', 'alter_loss', 'distance2', 'reordering_loss', 'corpus_gleu')
    progressbar = tqdm(total=args.eval_every, desc='start training.')

    # cache
    samples = 100
    tau = 1

    caches = Cache(args.max_cache, ['src', 'trg', 'dec', 'fer'])
    best_model = copy.deepcopy(model)   # used for decoding
    best_score = 0

    # start loop
    iters = offset
    train = iter(train)
    counters = 0

    while iters <= args.maximum_steps:

        iters += 1
        counters += 1

        batch = devol(next(train))

        # prepare inputs
        model.eval()
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks0, encoding, batch_size = model.quick_prepare(batch)
        _, src_len, hsize = encoding[0].size()
        trg_len = targets.size(1)

        # prepare parallel -- noisy sampling
        decoder_inputs, input_reorder, decoder_masks, _, pred_fer \
                        = model.prepare_initial(encoding, sources, source_masks0, input_masks,
                                                None, None, decoding=True, mode='reinforce',
                                                N=samples, tau=tau, return_samples=True)

        # repeating for decoding
        source_masks = source_masks0[:, None, :].expand(batch_size, samples,
                       src_len).contiguous().view(batch_size * samples, src_len)
        for i in range(len(encoding)):
            encoding[i] = encoding[i][:, None, :].expand(
            batch_size, samples, src_len, hsize).contiguous().view(batch_size * samples, src_len, hsize)

        # run decoding
        decoding, _, probs = best_model(encoding, source_masks, decoder_inputs, decoder_masks, decoding=True, return_probs=True)

        # compute GLEU score to select the best translation
        trg_output = best_model.output_decoding(('trg', targets[:, None, :].expand(batch_size,
                                                samples, trg_len).contiguous().view(batch_size * samples, trg_len)))
        dec_output = best_model.output_decoding(('trg', decoding))
        bleu_score = computeBLEU(dec_output, trg_output, corpus=False, tokenizer=tokenizer).contiguous().view(batch_size, samples).cuda(args.gpu)
        best_index = bleu_score.max(1)[1]

        def index_gather(data, index, samples):
            batch_size = index.size(0)
            data = data.contiguous().view(batch_size, samples, -1)  # batch x samples x dim
            index = index[:, None, None].expand(batch_size, 1, data.size(2))
            return data.gather(1, index)[:, 0, :]

        best_decoding, best_decoder_masks, best_fertilities = [index_gather(x, best_index, samples) for x in [decoding, decoder_masks, pred_fer]]
        caches.add([sources, targets, best_decoding, best_fertilities],
                    [source_masks0, target_masks, best_decoder_masks, source_masks0],
                    ['src', 'trg', 'dec', 'fer'])


        progressbar.update(1)
        progressbar.set_description('caching sentences={}/batches={}'.format(len(caches.cache), iters))


        if counters == args.eval_every:
            logger.info('build a new dataset from the caches')
            print(len(caches.cache))

            cache_data = ParallelDataset(examples=caches.cache,
                                        fields=[('src', SRC), ('trg', TRG), ('dec', TRG), ('fer', FER)])
            cache_iter = data.BucketIterator(cache_data, batch_sizes=2048, device=args.gpu, batch_size_fn=batch_size_fn)
            print('done')
            import sys;sys.exit(1)


        if False: # iters % args.eval_every == 0:
            progressbar.close()
            dev_metrics.reset()

            outputs_data = valid_model(model, dev, None if args.seq_dist else dev_metrics, teacher_model=None, print_out=True)

            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/GLEU_sentence_', dev_metrics.gleu, iters)
                writer.add_scalar('dev/Loss', dev_metrics.loss, iters)
                writer.add_scalar('dev/GLEU_corpus_', outputs_data['corpus_gleu'], iters)
                writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], iters)

            if not args.debug:
                best.accumulate(outputs_data['corpus_bleu'], outputs_data['corpus_gleu'], dev_metrics.gleu, dev_metrics.loss, iters)
                logger.info('the best model is achieved at {}, average greedy GLEU={}, corpus GLEU={}, corpus BLEU={}'.format(
                    best.i, best.gleu, best.corpus_gleu, best.corpus_bleu))

            logger.info('model:' + args.prefix + hp_str)

            # ---set-up a new progressor---
            progressbar = tqdm(total=args.eval_every, desc='start training.')



if args.mode == 'train':
    logger.info('starting training')
    train_model(model, train_real, dev_real, teacher_model)

elif args.mode == 'self':
    logger.info('starting self-training')
    self_improving_model(model, train_real, dev_real)

elif args.mode == 'test':
    logger.info('starting decoding from the pre-trained model, test...')

    names = ['dev.src.b{}={}.{}'.format(args.beam_size, args.load_from, args,fertility_mode),
            'dev.trg.b{}={}.{}'.format(args.beam_size, args.load_from, args,fertility_mode),
            'dev.dec.b{}={}.{}'.format(args.beam_size, args.load_from, args,fertility_mode)]
    decode_model(model, None, dev_real, evaluate=True, decoding_path=decoding_path if not args.no_write else None, names=names)

elif args.mode == 'test_noisy':
    logger.info('starting decoding from the pre-trained model, test...')

    names = ['dev.src.b{}={}.noise{}'.format(args.beam_size, args.load_from, args.beam_size),
            'dev.trg.b{}={}.noise{}'.format(args.beam_size, args.load_from, args.beam_size),
            'dev.dec.b{}={}.noise{}'.format(args.beam_size, args.load_from, args.beam_size),
            'dev.fer.b{}={}.noise{}'.format(args.beam_size, args.load_from, args.beam_size)]
    noisy_decode_model(model, dev_real, samples=args.beam_size, alpha=args.alpha, tau=args.temperature,
                        teacher_model=teacher_model, evaluate=True, decoding_path=decoding_path if not args.no_write else None,
                        names=names, saveall=True)
else:
    logger.info('starting decoding from the pre-trained model, build the course dataset...')
    names = ['src.b{}'.format(args.beam_size), 'trg.b{}'.format(args.beam_size), 'dec.b{}'.format(args.beam_size)]
    decode_model(model, train_real, dev_real, decoding_path=decoding_path if not args.no_write else None, names=names)

logger.info("done.")
