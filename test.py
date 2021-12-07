import matplotlib
import os
from argparse import ArgumentParser
import scipy
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable

from c_rnn_gan import Generator, Discriminator
import utils
from config import parsers
import matplotlib.pyplot as plt
DATA_DIR_TRN = '../../DATA/nsdi_DATA'
# DATA_DIR_VAL = '../../DATA/DATA_2000_10/test'
ARGS = parsers()
G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'
gloss_array = []
dloss_array = []

MAX_GRAD_NORM = 5.0
BATCH_SIZE = 32
MAX_EPOCHS = 500
L2_DECAY = 1.0

MAX_SEQ_LEN = 300

PERFORM_LOSS_CHECKING = False
FREEZE_G = False
FREEZE_D = False

NUM_DUMMY_TRN = 256   # 训练数据集总共256
NUM_DUMMY_VAL = 128   # 验证数据集总共128

EPSILON = 1e-40 # value to use to approximate zero (to prevent undefined results)


def write_log(log_values, model_name, log_dir="", log_type='loss', type_write='a'):
    if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    with open(log_dir+"/"+model_name+"_"+log_type+".txt", type_write) as f:
        f.write(','.join(log_values)+"\n")

def get_accuracy(logits_real, logits_gen):
    ''' Discriminator accuracy
    '''
    real_corrects = (logits_real > 0.5).sum()
    gen_corrects = (logits_gen < 0.5).sum()

    acc = (real_corrects + gen_corrects) / (len(logits_real) + len(logits_gen))
    return acc.item()


class GLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''
    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, a, logits_gen):
        logits_gen = torch.clamp(logits_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)

        return torch.mean(batch_loss)


class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''
    def __init__(self, label_smoothing=False):
        super(DLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_real, logits_gen):
        ''' Discriminator loss

        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator

        loss = -(ylog(p) + (1-y)log(1-p))

        '''
        logits_real = torch.clamp(logits_real, EPSILON, 1.0)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            "Label Smoothing"
            p_fake = torch.clamp((1 - logits_real), EPSILON, 1.0)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9*d_loss_real + 0.1*d_loss_fake

        logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen



        return torch.mean(batch_loss)


def control_grad(model, freeze=True):
    ''' Freeze/unfreeze optimization of model
    '''
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    else: # unfreeze
        for param in model.parameters():
            param.requires_grad = True


def check_loss(model, loss):
    ''' Check loss and control gradients if necessary
    '''
    control_grad(model['g'], freeze=False)
    control_grad(model['d'], freeze=False)

    if loss['d'] == 0.0 and loss['g'] == 0.0:
        print('Both G and D train loss are zero. Exiting.')
        return False
    elif loss['d'] == 0.0: # freeze D
        control_grad(model['d'], freeze=True)
    elif loss['g'] == 0.0: # freeze G
        control_grad(model['g'], freeze=True)
    # elif loss['g'] < 2.0 or loss['d'] < 2.0:
    #     control_grad(model['d'], freeze=True)
        if loss['g']*0.7 > loss['d']:
            control_grad(model['g'], freeze=True)

    return True

def get_random_batches(path, batch_size, gt=0):
    length = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]) - 1
    index_train = np.random.choice(int(length),batch_size,  False)
    data = []
    if gt==0:
        for index in index_train:
            tmp = np.load(path+'./{}.npy'.format(index))[:ARGS.seq_len]
            data.append(tmp)
        data = np.array(data)/151 - 1
        return data
    else:
        for index in index_train:
            tmp = np.load(path+'./{}.npy'.format(index))
            data.append(tmp)
        data = np.array(data)/151 - 1
        return data[:, :ARGS.seq_len], data[:, ARGS.seq_len:]

def run_training(model, optimizer, criterion, dataloader, ep, freeze_g=False, freeze_d=False):
    args = parsers()
    ''' Run single training epoch
    '''

    loss = {
        'g': 10.0,
        'd': 10.0
    }

    num_feats = model['g'].num_feats
    cuda = True if (torch.cuda.is_available() and args.GPU) else False
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    model['g'].train()
    model['d'].train()

    output_all = np.zeros((5, 50))
    for i, (batch_input, labels) in enumerate(dataloader):

        real_batch_sz = len(batch_input)
        batch_input = batch_input.type(torch.FloatTensor)  # 128,100

        # get initial states
        # each batch is independent i.e. not a continuation of previous batch
        # so we reset states for each batch
        # POSSIBLE IMPROVEMENT: next batch is continuation of previous batch
        g_states = model['g'].init_hidden(real_batch_sz)
        labels = Variable(labels.type(FloatTensor))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        if not freeze_d:
            optimizer['d'].zero_grad()
        # feed real and generated input to discriminator
        z = torch.empty([real_batch_sz, args.latent_dim]).normal_(0, 1)# random vector
        g_feats, _ = model['g'](z, g_states, batch_input.cuda())
        output_all = np.concatenate((output_all, g_feats.cpu().detach().numpy()), 0)
    output_all = output_all[5:]
    utils.mkr('../data_generated_{}/'.format(ep))
    for i in range(len(output_all)):
        tmp1 = output_all[i]
        tmp = np.array([int(np.round((x + 1) * 151).clip(0, 301)) for x in tmp1])
        tmp = np.hstack((tmp, tmp))
        np.save('../data_generated_{}/{}.npy'.format(ep, i), tmp)


def generate_sample(g_model, batches_done):
    ''' Sample from generator
    '''
    n_row = 5
    num_sample = n_row*n_row
    args = parsers()
    z = torch.empty([num_sample, args.latent_dim, args.num_feats]).normal_(0, 1) # random vector

    g_states = g_model.init_hidden(num_sample)
    LongTensor = torch.cuda.LongTensor
    gen_labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    gen_labels = Variable(LongTensor(gen_labels))

    g_feats, _ = g_model(z, g_states, gen_labels)
    gen_imgs = g_feats.cpu()
    gen_imgs = gen_imgs.detach().numpy()
    fig = plt.figure(figsize=(17, 17))
    for i_s in range(1, 26):
        ax = plt.subplot(5, 5, i_s)
        plt.plot(gen_imgs[i_s-1, :args.seq_len, 0], gen_imgs[i_s-1, :args.seq_len, 1], 2)
        value = max(gen_imgs[i_s-1, :args.seq_len, 0].max()-gen_imgs[i_s-1, :args.seq_len, 0].min(), gen_imgs[i_s-1, :args.seq_len, 1].max()-gen_imgs[i_s-1, :args.seq_len, 1].min())

        if value < 2.1092:
            r = 0
        elif value < 2.615:
            r = 1
        elif value < 2.963:
            r = 2
        elif value < 3.247:
            r = 3
        else:
            r = 4
        ax.set_title('{}-{:.2f}'.format(r, value))
    fig.suptitle('epoch:{}'.format(batches_done), fontsize=30)
    np.save(args.data_path+'/{}.npy'.format(batches_done), gen_imgs)
    plt.savefig(args.pics_path+'/{}_plot.jpg'.format(batches_done))
    plt.close()
    # save_image(g_feats.data, args.pics_path+ "%d.png" % batches_done, nrow=n_row, normalize=True)
    return


def main(args):
    ''' Training sequence
    '''
    epoch = 6000
    trn_dataloader = utils.load_data(args.batch_size, DATA_DIR_TRN)
    val_dataloader = utils.load_data(args.batch_size, DATA_DIR_TRN)

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available() and args.GPU
    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    model = {
        'g': Generator(num_feats=args.num_feats, use_cuda=train_on_gpu),
        'd': Discriminator(num_feats=args.num_feats, use_cuda=train_on_gpu)
    }

    generator_stat = torch.load('./Model2/g_epoch_{}.pt'.format(epoch))['model_state_dict']
    model['g'].load_state_dict(generator_stat)
    model['g'].cuda()

    optimizer = {
        # 'g': optim.SGD(model['g'].parameters(), G_LRN_RATE, weight_decay=L2_DECAY),
        'g': optim.Adam(model['g'].parameters(), args.g_lr),
        'd': optim.Adam(model['d'].parameters(), args.d_lr)
    }

    criterion = {
        'g': nn.MSELoss(reduction='sum'),
        'd': DLoss(label_smoothing=args.label_smoothing)
    }

    if train_on_gpu:
        model['g'].cuda()
        model['d'].cuda()

    # ---------------------
    #  Pre training
    # ---------------------

    if not args.no_pretraining:
        for ep in range(args.pretraining_epochs):
            model, trn_g_loss, trn_d_loss, trn_acc = \
                run_training(model, optimizer, criterion, trn_dataloader, ep, freeze_g=True)
            # val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

            print("Pretraining Epoch %d/%d\n"
                  "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
                  "############################################################" %
                  (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc))


        for ep in range(args.pretraining_epochs):
            model, trn_g_loss, trn_d_loss, trn_acc = \
                run_training(model, optimizer, criterion, trn_dataloader, ep, freeze_d=True)
            # val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

            print("Pretraining Epoch %d/%d\n"
                  "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
                  "############################################################" %
                  (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc))

    # ---------------------
    #  Training
    # ---------------------
    flag = False
    for ep in range(args.num_epochs):
        model, trn_g_loss, trn_d_loss, trn_acc = run_training(model, optimizer, criterion, trn_dataloader, epoch, freeze_d=flag)
        if args.freezing:
            if trn_acc > 95:
                flag = True
                print("Freeze D!")
            else:
                flag = False
        # generate_sample(model['g'], ep)
        # val_g_loss, val_d_loss, val_acc = run_validation(model, criterion, val_dataloader)

        # print("Epoch %d/%d\n"
        #       "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
        #       "\t[Validation] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
        #       "############################################################" %
        #       (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc,
        #        val_g_loss, val_d_loss, val_acc))
        print("Epoch %d/%d\n"
              "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
              "############################################################" %
              (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc))
        with open("./log.txt", 'a') as f:
            f.write("Epoch %d/%d\n"
              "\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n"
              "############################################################\n" %
              (ep+1, args.num_epochs, trn_g_loss, trn_d_loss, trn_acc))
        # sampling (to check if generator really learns)


if __name__ == "__main__":


    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size
    FREEZE_G = ARGS.freeze_g
    FREEZE_D = ARGS.freeze_d

    # utils.mkr(ARGS.Model_path)
    # utils.mkr(ARGS.data_path)
    # utils.mkr(ARGS.pics_path)
    # utils.mkr(ARGS.stats_path)

    # path_to_file = './log.txt'
    # if os.path.exists(path_to_file):
    #     os.remove("./log.txt")
    DATA_DIR_TRN = '../data'
    # DATA_DIR_VAL = '../../DATA/_DATA_{}_{}/test'.format(ARGS.time_length, ARGS.trace_sample_interval)
    print(ARGS)
    main(ARGS)
