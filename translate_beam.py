import os
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.serialization import default_restore_location

from seq2seq import models, utils
from seq2seq.data.dictionary import Dictionary
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq.beam import BeamSearch, BeamSearchNode


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', default=False, help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--data', default='assignments/03/prepared', help='path to data directory')
    parser.add_argument('--dicts', required=True, help='path to directory containing source and target dictionaries')
    parser.add_argument('--checkpoint-path', default='checkpoints_asg4/checkpoint_best.pt', help='path to the model file')
    parser.add_argument('--batch-size', default=None, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', default='model_translations.txt', type=str,
                        help='path to the output file destination')
    parser.add_argument('--max-len', default=100, type=int, help='maximum length of generated sequence')

    # Add beam search arguments
    parser.add_argument('--beam-size', default=5, type=int, required=True, help='number of hypotheses expanded in beam search')
    # alpha hyperparameter for length normalization (described as lp in https://arxiv.org/pdf/1609.08144.pdf equation 14)
    parser.add_argument('--alpha', default=0.0, type=float, help='alpha for softer length normalization')
    # use squared regularizer (as described in https://aclanthology.org/2020.emnlp-main.170.pdf, eq. 16)
    parser.add_argument('--regularizer', default=False, type=bool, help='whether to apply regularizer or not')
    # lambda hyperparameter for squared regularizer
    parser.add_argument('--lambda_', default=0.0, type=float, help='hyperparameter to control strength of regularization')
    # how many best hypotheses to return
    parser.add_argument('--n_best', default=1, type=int, help='number of hypotheses to return')

    return parser.parse_args()


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    args = args_loaded
    utils.init_logging(args)

    assert args.beam_size > 0, f"beam_size must be positive"
    assert args.n_best <= args.beam_size, f"n_best must be smaller than or equal to {args.beam_size}"
    
    # Load dictionaries
    src_dict = Dictionary.load(os.path.join(args.dicts, 'dict.{:s}'.format(args.source_lang)))
    logging.info('Loaded a source dictionary ({:s}) with {:d} words'.format(args.source_lang, len(src_dict)))
    tgt_dict = Dictionary.load(os.path.join(args.dicts, 'dict.{:s}'.format(args.target_lang)))
    logging.info('Loaded a target dictionary ({:s}) with {:d} words'.format(args.target_lang, len(tgt_dict)))

    # Load dataset
    test_dataset = Seq2SeqDataset(
        src_file=os.path.join(args.data, 'test.{:s}'.format(args.source_lang)),
        tgt_file=os.path.join(args.data, 'test.{:s}'.format(args.target_lang)),
        src_dict=src_dict, tgt_dict=tgt_dict)

    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=1, collate_fn=test_dataset.collater,
                                            batch_sampler=BatchSampler(test_dataset, 9999999,
                                                                        args.batch_size, 1, 0, shuffle=False,
                                                                        seed=args.seed))
    # Build model and criterion
    model = models.build_model(args, src_dict, tgt_dict)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {:s}'.format(args.checkpoint_path))
    progress_bar = tqdm(test_loader, desc='| Generation', leave=False)

    # Iterate over the test set
    all_hyps = {}
    for i, sample in enumerate(progress_bar):
        # 'src_tokens' shape: (batch_size, src_time_steps), batch_size over 1st dimension of src_tokens
        batch_size = sample['src_tokens'].shape[0] 
        # Create a beam search objects of the batch size for every input sentence (total = batch_size * beam_size)
        searches = [BeamSearch(args.beam_size, args.max_len - 1, tgt_dict.unk_idx) for i in range(batch_size)]

        with torch.no_grad():
            # Compute the encoder output 
            # 'src_embeddings' shape: (time_steps, batch_size, encoder_hidden_size) 
            # 'src_out': 'lstm_out' shape: (time_steps, batch_size, 2*encoder_hidden_size),
            #            'final_hidden_states' shape: (1, 2*encoder_hidden_size, batch_size), 
            #            'final_cell_states' shape: (1, 2*encoder_hidden_size, batch_size)
            # 'src_mask' shape: (batch_size, time_steps)
            encoder_out = model.encoder(sample['src_tokens'], sample['src_lengths'])

            # in rnns and lstms we have to feed in previously generated output into decoder
            # to initialize decoder hidden state on a 1st time step, we feed in a tensor containing ones "go_slice"
            # "go_slice" shape: (batch_size, 1) 
            go_slice = \
                torch.ones(sample['src_tokens'].shape[0], 1).fill_(tgt_dict.eos_idx).type_as(sample['src_tokens'])
            if args.cuda:
                go_slice = utils.move_to_cuda(go_slice)
            
            # Compute the decoder output at the first time step
            # 'decoder_out' shape: (batch_size, 1, vocab_size)
            decoder_out, _ = model.decoder(go_slice, encoder_out)

            # torch.topk returns the beam_size number of the largest elements along the last dimension (https://pytorch.org/docs/stable/generated/torch.topk.html)
            # meaning: returns args.beam_size+1 number of elements with the highest probability (softmax) over the whole vocab (decoder_out, vocab_size)
            # we want to keep one top candidate more than the desired beam_size to prevent generation of emtpy sequence(s)
            # <EOS> alone can have higher log-prob than log-probs of some other candidates and end up in topk, but the beam search stops at <EOS>
            # and then we get the output sequence that contains only <EOS>: empty sequence (see https://aclanthology.org/W18-6322.pdf)
            # by adding one to the initial beam_size, we ensure to get the desired number of fully generated sequeces
            # 'log_probs'/'next candidates' shape: (batch_size, 1, beam_size+1)
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)),
                                                    args.beam_size+1, dim=-1)

        # Create number of beam_size beam search nodes for every input sentence
        for i in range(batch_size):
            for j in range(args.beam_size):
                best_candidate = next_candidates[i, :, j]
                backoff_candidate = next_candidates[i, :, j+1]
                best_log_p = log_probs[i, :, j]
                backoff_log_p = log_probs[i, :, j+1]
                next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                log_p = log_p[-1]

                # Store the encoder_out information for the current input sentence and beam
                emb = encoder_out['src_embeddings'][:,i,:]
                lstm_out = encoder_out['src_out'][0][:,i,:]
                final_hidden = encoder_out['src_out'][1][:,i,:]
                final_cell = encoder_out['src_out'][2][:,i,:]
                try:
                    mask = encoder_out['src_mask'][i,:]
                except TypeError:
                    mask = None

                node = BeamSearchNode(searches[i], emb, lstm_out, final_hidden, final_cell,
                                    mask, torch.cat((go_slice[i], next_word)), log_p, 1)

                # since the log function is always monotonically increasing, getting the maximal log value ensures that
                # we are indeed getting the most probable candidate word associated with this value.
                # with squared regularizer we would like to increase the log probs such that during decoding we choose a candidate word with the highest log prob
                # but by putting the minus sign in front of the node ('score' in beam.py), we will achieve the same.
                # PriorityQueue returns the smallest value by default, so if we add the minus sign in front of each log prob (node), the lowest log prob will become the highest,
                # thus always returned first. https://docs.python.org/3/library/queue.html#queue.PriorityQueue + Martijn Pieters.
                # note that log probs are always negative, i.e. log_2(p=0.4) = -1.322, log_2(p=0.6) = -0.737 etc.,
                # before applying any of the steps described previously: square/minus sign are going to make log probs positive.
                if not args.regularizer:
                    searches[i].add(-node.eval(args.alpha), node)
                else:
                    searches[i].add(node.eval(args.alpha), node) 

        # Start generating further tokens until max sentence length reached
        for _ in range(args.max_len-1):

            # Get the current nodes to expand (total nodes = batch_size * beam_size)
            nodes = list(n[1] for s in searches for n in s.get_current_beams())
            print(nodes)
            if nodes == []:
                break # All beams ended in EOS

            # Reconstruct prev_words, encoder_out from current beam search nodes
            prev_words = torch.stack([node.sequence for node in nodes])
            encoder_out["src_embeddings"] = torch.stack([node.emb for node in nodes], dim=1)
            lstm_out = torch.stack([node.lstm_out for node in nodes], dim=1)
            final_hidden = torch.stack([node.final_hidden for node in nodes], dim=1)
            final_cell = torch.stack([node.final_cell for node in nodes], dim=1)
            encoder_out["src_out"] = (lstm_out, final_hidden, final_cell)
            try:
                encoder_out["src_mask"] = torch.stack([node.mask for node in nodes], dim=0)
            except TypeError:
                encoder_out["src_mask"] = None

            with torch.no_grad():
                # Compute the decoder output by feeding it the decoded sentence prefix 
                # 'decoder_out' shape: (beam_size * batch_size, time_step, vocab_size)
                decoder_out, _ = model.decoder(prev_words, encoder_out)

            
            # 'log_probs'/'next candidates' shape: (beam_size * batch_size, time_step, beam_size+1)
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)), args.beam_size+1, dim=-1)

            # Create number of beam_size next nodes for every current node
            for i in range(log_probs.shape[0]):
                for j in range(args.beam_size):
                    # pick a best and the next (backoff) candidate
                    best_candidate = next_candidates[i, :, j]
                    backoff_candidate = next_candidates[i, :, j+1]
                    best_log_p = log_probs[i, :, j]
                    backoff_log_p = log_probs[i, :, j+1]
                    # if best candidate is "UNK" next_word is a next (backoff) candidate 
                    next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
                    log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
                    log_p = log_p[-1]
                    next_word = torch.cat((prev_words[i][1:], next_word[-1:]))

                    # Get parent node and beam search object for corresponding sentence
                    node = nodes[i]
                    search = node.search

                    # we want to store the node as final if <EOS> is generated
                    # "add_final" adds a node that can contain <EOS> 
                    # in this way we kow that we have completed a hypothesis which won't be expanded further
                    # otherwise, beam search would continue 
                    if next_word[-1] == tgt_dict.eos_idx:
                        node = BeamSearchNode(
                            search, node.emb, node.lstm_out, node.final_hidden,
                            node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                            next_word)), node.logp, node.length
                            )
                        if not args.regularizer:
                            search.add_final(-node.eval(args.alpha), node)
                        else:
                            search.add_final(args.lambda_ * node.eval(args.alpha)**2, node)
                    # in case the node doesn't contain <EOS>, we want to add the node to the current nodes for the next iteration (continue of the search)
                    # "add" adds a node that can contain anything else except for <EOS>
                    else:
                        if not args.regularizer:
                            node = BeamSearchNode(
                                search, node.emb, node.lstm_out, node.final_hidden,
                                node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                next_word)), node.logp + log_p, node.length + 1
                                )
                            search.add(-node.eval(args.alpha), node) 
                        else:
                            node = BeamSearchNode(
                                search, node.emb, node.lstm_out, node.final_hidden,
                                node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                next_word)), node.logp - log_p, node.length + 1
                            )
                            search.add(args.lambda_ * node.eval(args.alpha)**2, node)

            # we discontinue further search for all other nodes except for the beam_size number of nodes with the lowest negative log prob 
            # also, excluding the paths that are already finished (with <EOS>).
            for search in searches:
                search.prune()

        # Segment into sentences
        # 'best_sents' shape: (batch_size, max_length) -> (batch_size * n_best, max_len)
        best_sents = torch.stack([s[1].sequence[1:].cpu() for s in search.get_best(args.n_best) for search in searches]).numpy()
        assert best_sents.shape[0] == batch_size * args.n_best

        output_sentences = [best_sents[row, :] for row in range(best_sents.shape[0])]
        
        # loop over all the sequences (num_sequences == batch_size * n_best) 
        # each sequence is of the length as passed to args.max_len
        # if <EOS> symbol idx > 0, we store indices up to the current idx as a single sentence in temp
        temp = list()
        for sent in output_sentences:
            first_eos = np.where(sent == tgt_dict.eos_idx)[0]
            if len(first_eos) > 0: 
                temp.append(sent[:first_eos[0]])
            else:
                temp.append(sent)
        output_sentences = temp

        # Convert arrays of indices into strings of words
        output_sentences = [[tgt_dict.string(sent) for sent in output_sentences[i+i:args.n_best+i+i]] for i in range(batch_size)]

        for ii, sent in enumerate(output_sentences):
            all_hyps[int(sample['id'].data[ii])] = sent

    # Write to file
    if args.output is not None:
        with open(args.output, 'w') as out_file:
            for sent_id in range(len(all_hyps.keys())):
                out_file.write(all_hyps[sent_id] + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
