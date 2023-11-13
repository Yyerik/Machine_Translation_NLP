#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way.
"""

from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open
import torch
import torch.nn as nn
import numpy as np

import matplotlib
# if you are running on the gradx/ugradx/ another cluster,
# you will need the following line
# if you run on a local machine, you can comment it out
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
import torch.utils.data as data

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to conflict with other people's jobs.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indices
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"),
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the languages based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class EncoderRNN(nn.Module):
    """the class for the encoder RNN
    """

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.batch_size=batch_size
        """Initialize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM. 
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        "*** YOUR CODE HERE ***"
        # word embedding
        self.embedding = nn.Embedding(input_size, hidden_size)
        # bi-directional LSTM
        self.biLSTM = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        "*** YOUR CODE HERE ***"
        xt = self.embedding(input)
        output, (hidden, c) = self.biLSTM(xt)

        return output, hidden, c

    def get_initial_hidden_state(self):
        return torch.zeros((self.batch_size, self.hidden_size), device=device), torch.zeros(
            (self.batch_size, self.hidden_size), device=device)


class AttnDecoderRNN(nn.Module):
    """the class for the decoder
    """

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        # word embedding
        self.embedding = nn.Embedding(output_size, hidden_size)
        # decoder LSTM
        self.lstm = nn.LSTM(3 * self.hidden_size, self.hidden_size)
        # attention and its weights
        self.Wa = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.Ua = nn.Parameter(torch.zeros(2 * hidden_size, hidden_size))
        self.ba = nn.Parameter(torch.ones(hidden_size))

        self.W_p = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.V_p = nn.Parameter(torch.zeros(hidden_size))
        self.D = 5
        # print(self.W_p)

    def forward(self, input, hidden, c, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.
        """

        "*** YOUR CODE HERE ***"
        
        batch_size = input.size()[1]
        embed = self.embedding(input.to(device))
        embed = self.dropout(embed)
        embed = embed.squeeze()

        
        attn_w = torch.zeros(batch_size, self.max_length, device=device)
        for i in range(self.max_length):
            out = encoder_outputs[i, :, :]
            #print(out.size())
            
            tmp = torch.tanh(torch.matmul(hidden, self.Wa) + torch.matmul(out, self.Ua)).squeeze(0)
            #print(tmp.shape,torch.mv(tmp, self.ba).shape)

            if batch_size == 1:
                tmp = tmp.view(batch_size, -1)

            # print(torch.mv(tmp[j, :, :], self.ba).size())
            attn_w[:, i] = torch.mv(tmp, self.ba)

        if len(hidden.shape) == 2:
            hidden = hidden.unsqueeze(0)
        p_t = self.max_length * F.sigmoid(torch.mv(torch.tanh(hidden @ self.W_p).squeeze(0), self.V_p))
        # print(p_t.unsqueeze(1)-torch.arange(0, self.max_length + 1).float().repeat(p_t.shape[0],1))
        # print(hidden)
        gaussian = torch.exp(
            (torch.arange(0, self.max_length).float().repeat(p_t.shape[0], 1).to(device) - p_t.unsqueeze(1)) ** 2 / (
                        self.D / 2) ** 2)

        attn_w = F.softmax(attn_w*gaussian, dim=1)

        attn_w = attn_w.view(batch_size, 1, self.max_length)
        encoder_outputs = encoder_outputs.view(batch_size, self.max_length, -1)
        attn_out = torch.matmul(attn_w, encoder_outputs)
        attn_out = attn_out.squeeze()
        # print(attn_out.size())
        hidden = hidden.view(1, batch_size, self.hidden_size)
        c = c.view(1, batch_size, self.hidden_size)
        if batch_size == 1:
            embed = embed.view(1, -1)
            attn_out = attn_out.view(1, -1)
        
        
        inputs = torch.cat((embed, attn_out), 1).view(1, batch_size, -1)

        output, (hidden, c) = self.lstm(inputs, (hidden, c))

        output = self.out(output[0])
        output = F.log_softmax(output, dim=1)
        
        return output, hidden, c, attn_w

    # def get_initial_hidden_state(self):
    # return torch.zeros(1, self.batch_size, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    # ip_len = input_tensor.size(1)
    tgt_len = target_tensor.size(0)

    
    loss = 0

    # for i in range(ip_len):

    optimizer.zero_grad()
    encoder_outputs, encoder_hidden, c = encoder(input_tensor)


    decoder_inputs = torch.tensor([[SOS_index for i in range(input_tensor.size()[1])]], device=device)
    decoder_hidden = encoder_hidden[1, :, :]
    # print(decoder_inputs.size())
    c = c[1, :, :]

    for t in range(tgt_len):
        #print(decoder_inputs.size(),decoder_hidden.size(), encoder_outputs.size())
        decoder_outputs, decoder_hidden, c, decoder_attention = decoder(
            decoder_inputs, decoder_hidden, c, encoder_outputs)
        
        loss += criterion(decoder_outputs, target_tensor[t])

        decoder_inputs = torch.zeros((1, decoder_inputs.shape[1]), dtype=torch.long)
        topv, topi = decoder_outputs.data.topk(1)
        for kk in range(decoder_outputs.shape[0]):
            teacher_force = random.random() < 0.5
            if not teacher_force:
                decoder_inputs[0, kk] = torch.LongTensor([[topi[kk, 0]]])
            else:
                decoder_inputs[0, kk] = torch.LongTensor([[target_tensor[t, kk]]])
        # decoder_inputs = target_tensor[t].view(1, -1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        loss = 0

        encoder_outputs, encoder_hidden, c = encoder(input_tensor)
        
        zeros_pad = torch.zeros(max_length - len(input_tensor), 1, encoder.hidden_size * 2, device=device)
        encoder_outputs = torch.cat((encoder_outputs, zeros_pad), 0)
        
        
        decoder_input = torch.tensor([[SOS_index]], device=device)
        decoder_input = decoder_input.view(1, -1)
        decoder_hidden = encoder_hidden[1, :, :]
        # print(encoder_outputs.size())
        # print(decoder_input.size())
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        c = c[1, :, :]

        for di in range(max_length):

            decoder_output, decoder_hidden, c, decoder_attention = decoder(
                decoder_input, decoder_hidden, c, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            decoder_input = decoder_input.view(1, -1)

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions, i):
    """visualize the attention mechanism. And save it to a file.
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    d = input_sentence.split(' ')
    input_len = len(d) + 1
    output_len = len(output_words) + 1
    attention_map = np.squeeze(attentions)[0:output_len, 0:input_len]

    pc = plt.figure(figsize=(4, 4))
    ax = pc.add_subplot(111)
    i = ax.imshow(attention_map, cmap='gray')



def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, i):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions, i)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', default=6, type=int, help='batch_size of mini-batch')
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=2000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=5000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    # print(src_vocab.n_words)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every
    max_length = MAX_LENGTH

    while iter_num  < args.n_iters:
        if iter_num % 1000 == 0:
            print(iter_num)
        iter_num += 1
        sample = 0
        while sample < args.batch_size:
            sample += 1
            training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
            input_tensor_now = training_pair[0]
            target_tensor_now = training_pair[1]
            zeros_pad = torch.zeros(max_length - len(input_tensor_now), 1).type(torch.LongTensor).to(device)
            input_tensor_now = torch.cat((input_tensor_now, zeros_pad), 0)
            zeros_pad = torch.zeros(max_length - len(target_tensor_now), 1).type(torch.LongTensor).to(device)
            target_tensor_now = torch.cat((target_tensor_now, zeros_pad), 0)
            if sample == 1:
                input_tensor = input_tensor_now.view(max_length, -1)
                target_tensor = target_tensor_now.view(max_length, -1)
            else:
                input_tensor_now = input_tensor_now.view(max_length, -1)
                input_tensor = torch.cat((input_tensor, input_tensor_now), 1)
                target_tensor_now = target_tensor_now.view(max_length, -1)
                target_tensor = torch.cat((target_tensor, target_tensor_now), 1)
        # input_tensor=torch.tensor(B_input,dtype=torch.long, device=device).view(-1,len(B_input))
        # target_tensor = torch.tensor(B_target, dtype=torch.long, device=device).view(-1,len(B_target))

        # print(input_tensor.size(),target_tensor.size())

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (it:%d /n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab, 'a')
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab, 'b')
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab, 'c')
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab, 'd')


if __name__ == '__main__':
    main()