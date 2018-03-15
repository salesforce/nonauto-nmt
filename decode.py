# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
import os
import torch
import numpy as np
import time

from torch.nn import functional as F
from torch.autograd import Variable

from tqdm import tqdm, trange
from model import Transformer, FastTransformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset
from utils import Metrics, Best, computeGLEU, computeBLEU, Batch, masked_sort, computeGroupBLEU
from time import gmtime, strftime


tokenizer = lambda x: x.replace('@@ ', '').split()
def cutoff(s, t):
    for i in range(len(s), 0, -1):
        if s[i-1] != t:
            return s[:i]
    print(s)
    raise IndexError


def decode_model(args, model, dev, teacher_model=None, evaluate=True,
                decoding_path=None, names=None, maxsteps=None):

    args.logger.info("decoding with {}, f_size={}, beam_size={}, alpha={}".format(args.decode_mode, args.f_size, args.beam_size, args.alpha))
    dev.train = False  # make iterator volatile=True

    if maxsteps is None:
        progressbar = tqdm(total=sum([1 for _ in dev]), desc='start decoding')
    else:
        progressbar = tqdm(total=maxsteps, desc='start decoding')

    model.eval()
    if teacher_model is not None:
        assert (args.f_size * args.beam_size > 1), 'multiple samples are essential.'
        teacher_model.eval()

    if decoding_path is not None:
        handles = [open(os.path.join(decoding_path, name), 'w') for name in names]

    corpus_size = 0
    src_outputs, trg_outputs, dec_outputs, timings = [], [], [], []
    decoded_words, target_words, decoded_info = 0, 0, 0

    attentions = None
    pad_id = model.decoder.field.vocab.stoi['<pad>']
    eos_id = model.decoder.field.vocab.stoi['<eos>']

    curr_time = 0
    for iters, dev_batch in enumerate(dev):

        if iters > maxsteps:
            args.logger.info('complete {} steps of decoding'.format(maxsteps))
            break

        start_t = time.time()
        # encoding
        inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(dev_batch)

        if args.model is Transformer:
            # decoding from the Transformer

            decoder_inputs, decoder_masks = inputs, input_masks
            decoding = model(encoding, source_masks, decoder_inputs, decoder_masks,
                            beam=args.beam_size, alpha=args.alpha, decoding=True, feedback=attentions)
        else:
            # decoding from the FastTransformer

            if teacher_model is not None:
                encoding_teacher = teacher_model.encoding(sources, source_masks)

            decoder_inputs, input_reorder, decoder_masks, _, fertility = \
                    model.prepare_initial(encoding, sources, source_masks, input_masks, None, mode=args.decode_mode, N=args.f_size)
            batch_size, src_len, hsize = encoding[0].size()
            trg_len = targets.size(1)

            if args.f_size > 1:
                source_masks = source_masks[:, None, :].expand(batch_size, args.f_size, src_len)
                source_masks = source_masks.contiguous().view(batch_size * args.f_size, src_len)
                for i in range(len(encoding)):
                    encoding[i] = encoding[i][:, None, :].expand(
                    batch_size, args.f_size, src_len, hsize).contiguous().view(batch_size * args.f_size, src_len, hsize)
            decoding = model(encoding, source_masks, decoder_inputs, decoder_masks, beam=args.beam_size, decoding=True, feedback=attentions)
            total_size = args.beam_size * args.f_size

            # print(fertility.data.sum() - decoder_masks.sum())
            # print(fertility.data.sum() * args.beam_size - (decoding.data != 1).long().sum())
            if total_size > 1:
                if args.beam_size > 1:
                    source_masks = source_masks[:, None, :].expand(batch_size * args.f_size,
                        args.beam_size, src_len).contiguous().view(batch_size * total_size, src_len)
                    fertility = fertility[:, None, :].expand(batch_size * args.f_size,
                        args.beam_size, src_len).contiguous().view(batch_size * total_size, src_len)
                    # fertility = model.apply_mask(fertility, source_masks, -1)

                if teacher_model is not None:  # use teacher model to re-rank the translation
                    decoder_masks = teacher_model.prepare_masks(decoding)

                    for i in range(len(encoding_teacher)):
                        encoding_teacher[i] = encoding_teacher[i][:, None, :].expand(
                            batch_size,  total_size, src_len, hsize).contiguous().view(
                            batch_size * total_size, src_len, hsize)

                    student_inputs,  _ = teacher_model.prepare_inputs( dev_batch, decoding, decoder_masks)
                    student_targets, _ = teacher_model.prepare_targets(dev_batch, decoding, decoder_masks)
                    out, probs = teacher_model(encoding_teacher, source_masks, student_inputs, decoder_masks, return_probs=True, decoding=False)
                    _, teacher_loss = model.batched_cost(student_targets, decoder_masks, probs, batched=True)  # student-loss (MLE)

                    # reranking the translation
                    teacher_loss = teacher_loss.view(batch_size, total_size)
                    decoding = decoding.view(batch_size, total_size, -1)
                    fertility = fertility.view(batch_size, total_size, -1)
                    lp = decoder_masks.sum(1).view(batch_size, total_size) ** (1 - args.alpha)
                    teacher_loss = teacher_loss * Variable(lp)

                    # selected index
                    selected_idx = (-teacher_loss).topk(1, 1)[1]   # batch x 1
                    decoding = decoding.gather(1, selected_idx[:, :, None].expand(batch_size, 1, decoding.size(-1)))[:, 0, :]
                    fertility = fertility.gather(1, selected_idx[:, :, None].expand(batch_size, 1, fertility.size(-1)))[:, 0, :]

                else:   # (cheating, re-rank by sentence-BLEU score)

                    # compute GLEU score to select the best translation
                    trg_output = model.output_decoding(('trg', targets[:, None, :].expand(batch_size,
                                                        total_size, trg_len).contiguous().view(batch_size * total_size, trg_len)))
                    dec_output = model.output_decoding(('trg', decoding))
                    bleu_score = computeBLEU(dec_output, trg_output, corpus=False, tokenizer=tokenizer).contiguous().view(batch_size, total_size)
                    bleu_score = bleu_score.cuda(args.gpu)
                    selected_idx = bleu_score.max(1)[1]

                    decoding = decoding.view(batch_size, total_size, -1)
                    fertility = fertility.view(batch_size, total_size, -1)
                    decoding = decoding.gather(1, selected_idx[:, None, None].expand(batch_size, 1, decoding.size(-1)))[:, 0, :]
                    fertility = fertility.gather(1, selected_idx[:, None, None].expand(batch_size, 1, fertility.size(-1)))[:, 0, :]

                    # print(fertility.data.sum() - (decoding.data != 1).long().sum())
                    assert (fertility.data.sum() - (decoding.data != 1).long().sum() == 0), 'fer match decode'


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
                print(s, file=handles[0], flush=True)
                print(t, file=handles[1], flush=True)
                print(d, file=handles[2], flush=True)

            if args.model is FastTransformer:
                with torch.cuda.device_of(fertility):
                    fertility = fertility.data.tolist()
                    for f in fertility:
                        f = ' '.join([str(fi) for fi in cutoff(f, 0)])
                        print(f, file=handles[3], flush=True)

        progressbar.update(1)
        progressbar.set_description('finishing sentences={}/batches={}, speed={} sec/batch'.format(corpus_size, iters, curr_time / (1 + iters)))

    if evaluate:
        corpus_gleu = computeGLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
        corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
        args.logger.info("The dev-set corpus GLEU = {}".format(corpus_gleu))
        args.logger.info("The dev-set corpus BLEU = {}".format(corpus_bleu))

        # computeGroupBLEU(dec_outputs, trg_outputs, tokenizer=tokenizer)
        # torch.save([src_outputs, trg_outputs, dec_outputs, timings], './space/data.pt')
