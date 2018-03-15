# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
import torch
import numpy as np
import revtok
import os

from torch.autograd import Variable
from torchtext import data, datasets
from nltk.translate.gleu_score import sentence_gleu, corpus_gleu
from nltk.translate.bleu_score import corpus_bleu
from contextlib import ExitStack
from collections import OrderedDict

INF = 1e10
TINY = 1e-9
def computeGLEU(outputs, targets, corpus=False, tokenizer=None):
    if tokenizer is None:
        tokenizer = revtok.tokenize

    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]

    if not corpus:
        return torch.Tensor([sentence_gleu(
            [t],  o) for o, t in zip(outputs, targets)])
    return corpus_gleu([[t] for t in targets], [o for o in outputs])

def computeBLEU(outputs, targets, corpus=False, tokenizer=None):
    if tokenizer is None:
        tokenizer = revtok.tokenize

    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]

    if not corpus:
        return torch.Tensor([sentence_gleu(
            [t],  o) for o, t in zip(outputs, targets)])
    return corpus_bleu([[t] for t in targets], [o for o in outputs], emulate_multibleu=True)

def computeGroupBLEU(outputs, targets, tokenizer=None, bra=10, maxmaxlen=80):
    if tokenizer is None:
        tokenizer = revtok.tokenize

    outputs = [tokenizer(o) for o in outputs]
    targets = [tokenizer(t) for t in targets]
    maxlens = max([len(t) for t in targets])
    print(maxlens)
    maxlens = min([maxlens, maxmaxlen])
    nums = int(np.ceil(maxlens / bra))
    outputs_buckets = [[] for _ in range(nums)]
    targets_buckets = [[] for _ in range(nums)]
    for o, t in zip(outputs, targets):
        idx = len(o) // bra
        if idx >= len(outputs_buckets):
            idx = -1
        outputs_buckets[idx] += [o]
        targets_buckets[idx] += [t]

    for k in range(nums):
        print(corpus_bleu([[t] for t in targets_buckets[k]], [o for o in outputs_buckets[k]], emulate_multibleu=True))


# load the dataset + reversible tokenization
class NormalField(data.Field):

    def reverse(self, batch):
        if not self.batch_first:
            batch.t_()

        with torch.cuda.device_of(batch):
            batch = batch.tolist()

        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch] # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch] # trim past frst eos
        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [" ".join(filter(filter_special, ex)) for ex in batch]
        return batch

class NormalTranslationDataset(datasets.TranslationDataset):
    """Defines a dataset for machine translation."""

    def __init__(self, path, exts, fields, load_dataset=False, prefix='', **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        if load_dataset and (os.path.exists(path + '.processed.{}.pt'.format(prefix))):
            examples = torch.load(path + '.processed.{}.pt'.format(prefix))
        else:
            examples = []
            with open(src_path) as src_file, open(trg_path) as trg_file:
                for src_line, trg_line in zip(src_file, trg_file):
                    src_line, trg_line = src_line.strip(), trg_line.strip()
                    if src_line != '' and trg_line != '':
                        examples.append(data.Example.fromlist(
                            [src_line, trg_line], fields))
            if load_dataset:
                torch.save(examples, path + '.processed.{}.pt'.format(prefix))

        super(datasets.TranslationDataset, self).__init__(examples, fields, **kwargs)

class TripleTranslationDataset(datasets.TranslationDataset):
    """Define a triple-translation dataset: src, trg, dec(output of a pre-trained teacher)"""

    def __init__(self, path, exts, fields, load_dataset=False, prefix='', **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]), ('dec', fields[2])]

        src_path, trg_path, dec_path = tuple(os.path.expanduser(path + x) for x in exts)
        if load_dataset and (os.path.exists(path + '.processed.{}.pt'.format(prefix))):
            examples = torch.load(path + '.processed.{}.pt'.format(prefix))
        else:
            examples = []
            with open(src_path) as src_file, open(trg_path) as trg_file, open(dec_path) as dec_file:
                for src_line, trg_line, dec_line in zip(src_file, trg_file, dec_file):
                    src_line, trg_line, dec_line = src_line.strip(), trg_line.strip(), dec_line.strip()
                    if src_line != '' and trg_line != '' and dec_line != '':
                        examples.append(data.Example.fromlist(
                            [src_line, trg_line, dec_line], fields))
            if load_dataset:
                torch.save(examples, path + '.processed.{}.pt'.format(prefix))

        super(datasets.TranslationDataset, self).__init__(examples, fields, **kwargs)

class ParallelDataset(datasets.TranslationDataset):
    """ Define a N-parallel dataset: supports abitriry numbers of input streams"""

    def __init__(self, path=None, exts=None, fields=None,
                load_dataset=False, prefix='', examples=None, **kwargs):

        if examples is None:
            assert len(exts) == len(fields), 'N parallel dataset must match'
            self.N = len(fields)

            paths = tuple(os.path.expanduser(path + x) for x in exts)
            if load_dataset and (os.path.exists(path + '.processed.{}.pt'.format(prefix))):
                examples = torch.load(path + '.processed.{}.pt'.format(prefix))
            else:
                examples = []
                with ExitStack() as stack:
                    files = [stack.enter_context(open(fname)) for fname in paths]
                    for lines in zip(*files):
                        lines = [line.strip() for line in lines]
                        if not any(line == '' for line in lines):
                            examples.append(data.Example.fromlist(lines, fields))
                if load_dataset:
                    torch.save(examples, path + '.processed.{}.pt'.format(prefix))

        super(datasets.TranslationDataset, self).__init__(examples, fields, **kwargs)

class Metrics:

    def __init__(self, name, *metrics):
        self.count = 0
        self.metrics = OrderedDict((metric, 0) for metric in metrics)
        self.name = name

    def accumulate(self, count, *values, print_iter=None):
        self.count += count
        if print_iter is not None:
            print(print_iter, end=' ')
        for value, metric in zip(values, self.metrics):
            if isinstance(value, torch.autograd.Variable):
                value = value.data
            if torch.is_tensor(value):
                with torch.cuda.device_of(value):
                    value = value.cpu()
                value = value.float().mean()

            if print_iter is not None:
                print('%.3f' % value, end=' ')
            self.metrics[metric] += value * count
        if print_iter is not None:
            print()
        return values[0] # loss

    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key] / (self.count + 1e-9)
        raise AttributeError

    def __repr__(self):
        return (f"{self.name}: " +
                ', '.join(f'{metric}: {getattr(self, metric):.3f}'
                          for metric, value in self.metrics.items()
                          if value is not 0))

    def tensorboard(self, expt, i):
        for metric in self.metrics:
            value = getattr(self, metric)
            if value != 0:
                expt.add_scalar_value(f'{self.name}_{metric}', value, step=i)

    def reset(self):
        self.count = 0
        self.metrics.update({metric: 0 for metric in self.metrics})

class Best:

    def __init__(self, cmp_fn, *metrics, model=None, opt=None, path='', gpu=0):
        self.cmp_fn = cmp_fn
        self.model = model
        self.opt = opt
        self.path = path + '.pt'
        self.metrics = OrderedDict((metric, None) for metric in metrics)
        self.gpu = gpu

    def accumulate(self, cmp_value, *other_values):

        with torch.cuda.device(self.gpu):
            cmp_metric, best_cmp_value = list(self.metrics.items())[0]
            if best_cmp_value is None or self.cmp_fn(
                    best_cmp_value, cmp_value) == cmp_value:

                self.metrics[cmp_metric] = cmp_value
                self.metrics.update({metric: value for metric, value in zip(
                    list(self.metrics.keys())[1:], other_values)})

                open(self.path + '.temp', 'w')
                if self.model is not None:
                    torch.save(self.model.state_dict(), self.path)
                if self.opt is not None:
                    torch.save([self.i, self.opt.state_dict()], self.path + '.states')
                os.remove(self.path + '.temp')


    def __getattr__(self, key):
        if key in self.metrics:
            return self.metrics[key]
        raise AttributeError

    def __repr__(self):
        return ("BEST: " +
                ', '.join(f'{metric}: {getattr(self, metric):.3f}'
                        for metric, value in self.metrics.items()
                        if value is not None))

class CacheExample(data.Example):

    @classmethod
    def fromsample(cls, data_lists, names):
        ex = cls()
        for data, name in zip(data_lists, names):
            setattr(ex, name, data)
        return ex


class Cache:

    def __init__(self, size=10000, fileds=["src", "trg"]):
        self.cache = []
        self.maxsize = size

    def demask(self, data, mask):
        with torch.cuda.device_of(data):
            data = [d[:l] for d, l in zip(data.data.tolist(), mask.sum(1).long().tolist())]
        return data

    def add(self, data_lists, masks, names):
        data_lists = [self.demask(d, m) for d, m in zip(data_lists, masks)]
        for data in zip(*data_lists):
            self.cache.append(CacheExample.fromsample(data, names))

        if len(self.cache) >= self.maxsize:
            self.cache = self.cache[-self.maxsize:]


class Batch:
    def __init__(self, src=None, trg=None, dec=None):
        self.src, self.trg, self.dec = src, trg, dec

def masked_sort(x, mask, dim=-1):
    x.data += ((1 - mask) * INF).long()
    y, i = torch.sort(x, dim)
    y.data *= mask.long()
    return y, i

def unsorted(y, i, dim=-1):
    z = Variable(y.data.new(*y.size()))
    z.scatter_(dim, i, y)
    return z


def merge_cache(decoding_path, names0, last_epoch=0, max_cache=20):
    file_lock = open(decoding_path + '/_temp_decode', 'w')

    for name in names0:
        filenames = []
        for i in range(max_cache):
            filenames.append('{}/{}.ep{}'.format(decoding_path, name, last_epoch - i))
            if (last_epoch - i) <= 0:
                break
        code = 'cat {} > {}.train.{}'.format(" ".join(filenames), '{}/{}'.format(decoding_path, name), last_epoch)
        os.system(code)
    os.remove(decoding_path + '/_temp_decode')
