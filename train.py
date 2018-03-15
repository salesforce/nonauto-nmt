# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
import torch
import numpy as np
import math

from torch.nn import functional as F
from torch.autograd import Variable

from tqdm import tqdm, trange
from model import Transformer, FastTransformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset
from utils import Metrics, Best, computeGLEU, computeBLEU, Batch, masked_sort, computeGroupBLEU
from time import gmtime, strftime

# helper functions
def register_nan_checks(m):
    def check_grad(module, grad_input, grad_output):
        if any(np.any(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
            print('NaN gradient in ' + type(module).__name__)
            1/0
    m.apply(lambda module: module.register_backward_hook(check_grad))

def export(x):
    try:
        with torch.cuda.device_of(x):
            return x.data.cpu().float().mean()
    except Exception:
        return 0

def devol(batch):
    new_batch = copy.copy(batch)
    new_batch.src = Variable(batch.src.data, volatile=True)
    return new_batch

tokenizer = lambda x: x.replace('@@ ', '').split()

def valid_model(args, model, dev, dev_metrics=None, distillation=False,
                print_out=False, teacher_model=None):
    print_seqs = ['[sources]', '[targets]', '[decoded]', '[fertili]', '[origind]']
    trg_outputs, dec_outputs = [], []
    outputs = {}

    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    for j, dev_batch in enumerate(dev):
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks, \
        encoding, batch_size = model.quick_prepare(dev_batch, distillation)

        decoder_inputs, input_reorder, fertility_cost = inputs, None, None
        if type(model) is FastTransformer:
            decoder_inputs, input_reorder, decoder_masks, fertility_cost, pred_fertility = \
                model.prepare_initial(encoding, sources, source_masks, input_masks, None, mode='argmax')
        else:
            decoder_masks = input_masks

        decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks, decoding=True, return_probs=True)
        dev_outputs = [('src', sources), ('trg', targets), ('trg', decoding)]
        if type(model) is FastTransformer:
            dev_outputs += [('src', input_reorder)]
        dev_outputs = [model.output_decoding(d) for d in  dev_outputs]
        gleu = computeGLEU(dev_outputs[2], dev_outputs[1], corpus=False, tokenizer=tokenizer)

        if print_out:
            for k, d in enumerate(dev_outputs):
                args.logger.info("{}: {}".format(print_seqs[k], d[0]))
            args.logger.info('------------------------------------------------------------------')

        if teacher_model is not None:  # teacher is Transformer, student is FastTransformer
            inputs_student, _, targets_student, _, _, _, encoding_teacher, _ \
                                = teacher_model.quick_prepare(dev_batch, False, decoding, decoding, input_masks, target_masks, source_masks)
            teacher_real_loss = teacher_model.cost(targets, target_masks,
                                out=teacher_model(encoding_teacher, source_masks, inputs, input_masks))
            teacher_fake_out   = teacher_model(encoding_teacher, source_masks, inputs_student, input_masks)
            teacher_fake_loss  = teacher_model.cost(targets_student, target_masks, out=teacher_fake_out)
            teacher_alter_loss = teacher_model.cost(targets, target_masks, out=teacher_fake_out)

        trg_outputs += dev_outputs[1]
        dec_outputs += dev_outputs[2]

        if dev_metrics is not None:
            values = [0, gleu]
            if teacher_model is not None:
                values  += [teacher_real_loss, teacher_fake_loss,
                            teacher_real_loss - teacher_fake_loss,
                            teacher_alter_loss,
                            teacher_alter_loss - teacher_fake_loss]
            if fertility_cost is not None:
                values += [fertility_cost]

            dev_metrics.accumulate(batch_size, *values)

    corpus_gleu = computeGLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
    corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
    outputs['corpus_gleu'] = corpus_gleu
    outputs['corpus_bleu'] = corpus_bleu
    if dev_metrics is not None:
        args.logger.info(dev_metrics)

    args.logger.info("The dev-set corpus GLEU = {}".format(corpus_gleu))
    args.logger.info("The dev-set corpus BLEU = {}".format(corpus_bleu))
    return outputs


def train_model(args, model, train, dev, teacher_model=None, save_path=None, maxsteps=None):

    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('./runs/{}'.format(args.prefix+args.hp_str))

    # optimizer
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
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
    if save_path is None:
        save_path = args.model_name

    best = Best(max, 'corpus_bleu', 'corpus_gleu', 'gleu', 'loss', 'i', model=model, opt=opt, path=save_path, gpu=args.gpu)
    train_metrics = Metrics('train', 'loss', 'real', 'fake')
    dev_metrics = Metrics('dev', 'loss', 'gleu', 'real_loss', 'fake_loss', 'distance', 'alter_loss', 'distance2', 'fertility_loss', 'corpus_gleu')
    progressbar = tqdm(total=args.eval_every, desc='start training.')

    for iters, batch in enumerate(train):

        iters += offset

        if iters % args.save_every == 0:
            args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
                torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(args.model_name, iters))

        if iters % args.eval_every == 0:
            progressbar.close()
            dev_metrics.reset()

            if args.distillation:
                outputs_course = valid_model(args, model, dev, dev_metrics, distillation=True, teacher_model=None)

            outputs_data = valid_model(args, model, dev, None if args.distillation else dev_metrics, teacher_model=None, print_out=True)
            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/GLEU_sentence_', dev_metrics.gleu, iters)
                writer.add_scalar('dev/Loss', dev_metrics.loss, iters)
                writer.add_scalar('dev/GLEU_corpus_', outputs_data['corpus_gleu'], iters)
                writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], iters)

                if args.distillation:
                    writer.add_scalar('dev/GLEU_corpus_dis', outputs_course['corpus_gleu'], iters)
                    writer.add_scalar('dev/BLEU_corpus_dis', outputs_course['corpus_bleu'], iters)

            if not args.debug:
                best.accumulate(outputs_data['corpus_bleu'], outputs_data['corpus_gleu'], dev_metrics.gleu, dev_metrics.loss, iters)
                args.logger.info('the best model is achieved at {}, average greedy GLEU={}, corpus GLEU={}, corpus BLEU={}'.format(
                    best.i, best.gleu, best.corpus_gleu, best.corpus_bleu))
            args.logger.info('model:' + args.prefix + args.hp_str)

            # ---set-up a new progressor---
            progressbar = tqdm(total=args.eval_every, desc='start training.')


        if maxsteps is None:
            maxsteps = args.maximum_steps

        if iters > maxsteps:
            args.logger.info('reach the maximum updating steps.')
            break


        # --- training --- #
        model.train()
        def get_learning_rate(i, lr0=0.1, disable=False):
            if not disable:
                return lr0 * 10 / math.sqrt(args.d_model) * min(
                        1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
            return 0.00002
        opt.param_groups[0]['lr'] = get_learning_rate(iters + 1, disable=args.disable_lr_schedule)
        opt.zero_grad()

        # prepare the data
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks,\
        encoding, batch_size = model.quick_prepare(batch, args.distillation)
        input_reorder, fertility_cost, decoder_inputs = None, None, inputs
        batch_fer = batch.fer_dec if args.distillation else batch.fer

        #print(input_masks.size(), target_masks.size(), input_masks.sum())

        if type(model) is FastTransformer:
            inputs, input_reorder, input_masks, fertility_cost = model.prepare_initial(encoding, sources, source_masks, input_masks, batch_fer)


        # Maximum Likelihood Training
        if not args.finetuning:
            loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks))
            if args.fertility:
                loss += fertility_cost

        else:
            # finetuning:

            # loss_student (MLE)
            if not args.fertility:
                decoding, out, probs = model(encoding, source_masks, inputs, input_masks, return_probs=True, decoding=True)
                loss_student = model.batched_cost(targets, target_masks, probs)  # student-loss (MLE)
                decoder_masks = input_masks

            else: # Note that MLE and decoding has different translations. We need to run the same code twice
                # truth
                decoding, out, probs = model(encoding, source_masks, inputs, input_masks, decoding=True, return_probs=True)
                loss_student = model.cost(targets, target_masks, out=out)
                decoder_masks = input_masks

                # baseline
                decoder_inputs_b, _, decoder_masks_b, _, _ = model.prepare_initial(encoding, sources, source_masks, input_masks, None, mode='mean')
                decoding_b, out_b, probs_b = model(encoding, source_masks, decoder_inputs_b, decoder_masks_b, decoding=True, return_probs=True)  # decode again

                # reinforce
                decoder_inputs_r, _, decoder_masks_r, _, _ = model.prepare_initial(encoding, sources, source_masks, input_masks, None, mode='reinforce')
                decoding_r, out_r, probs_r = model(encoding, source_masks, decoder_inputs_r, decoder_masks_r, decoding=True, return_probs=True)  # decode again

            if args.fertility:
                loss_student += fertility_cost

            # loss_teacher (RKL+REINFORCE)
            teacher_model.eval()
            if not args.fertility:
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
                rewards = rewards - rewards.mean()
                rewards = rewards.expand_as(source_masks)
                rewards = rewards * source_masks

                model.predictor.saved_fertilities.reinforce(0.1 * rewards.contiguous().view(-1, 1))
                loss = (1 - args.beta1) * loss_teacher + args.beta1 * loss_student   # detect reinforce


        # accmulate the training metrics
        train_metrics.accumulate(batch_size, loss, print_iter=None)
        train_metrics.reset()

        # train the student
        if args.finetuning and args.fertility:
            torch.autograd.backward((loss, model.predictor.saved_fertilities),
                                    (torch.ones(1).cuda(loss.get_device()), None))
        else:
            loss.backward()
        opt.step()

        info = 'training step={}, loss={:.3f}, lr={:.5f}'.format(iters, export(loss), opt.param_groups[0]['lr'])
        if args.finetuning:
            info += '| NA:{:.3f}, AR:{:.3f}'.format(export(loss_student), export(loss_teacher))
            if args.fertility:
                info += '| RL: {:.3f}'.format(export(rewards.mean()))

        if args.fertility:
            info += '| RE:{:.3f}'.format(export(fertility_cost))


        if args.tensorboard and (not args.debug):
            writer.add_scalar('train/Loss', export(loss), iters)

        progressbar.update(1)
        progressbar.set_description(info)
