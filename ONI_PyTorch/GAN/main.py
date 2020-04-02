import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys
sys.path.append('..')
import model_resnet_ONI
import model_ONI
import model_resnet
import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

import functools
import inception_utils
import utils
import extension as my

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--z_dim', type=int, default=128)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--disc_iters', type=int, default=5)
parser.add_argument('--beta1', type=float, default=0)
parser.add_argument('--beta2', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=1)


parser.add_argument('--T', type=int, default=5)
parser.add_argument('--NScale', type=float, default=1)

my.visualization.add_arguments(parser)
args = parser.parse_args()

torch.set_num_threads(6)
utils.seed_rng(args.seed)
#BEGIN: visulization  and logger configurations for debug------
exp_Name = utils.name_from_config(args)
print('-----------ExpName:', exp_Name)
log_path = os.path.join('results', exp_Name)
os.makedirs(log_path, exist_ok=True)
logger = my.logger.setting('log.txt', log_path)
vis = my.visualization.setting(args, exp_Name, {
                               'lossD_true': 'loss of D with true input',
                               'lossD_fake': 'loss of D with fake input',
                               'lossG': 'loss of G',
                               'IS': 'Inception score',
                               'FID_train': 'FID score wrt train',
                               'FID_test': 'FID score wrt test'
                              })

#END: visulization  and logger configurations for debug------

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data/', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

Z_dim = args.z_dim
#number of updates to discriminator for every update to generator 
disc_iters = args.disc_iters

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
model_table = {
        'resnet': model_resnet,
        'resnet_ONI': model_resnet_ONI,
        'dcgan': model,
        'dcgan_ONI': model_ONI
        }

discriminator = model_table[args.model].Discriminator(args).cuda()
generator = model_table[args.model].Generator(args).cuda()

print(generator)
print(discriminator)
optim_disc = optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=args.lr, betas=(args.beta1,args.beta2))
optim_gen  = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# Seed RNG

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
get_inception_metrics = inception_utils.prepare_inception_metrics('C10', False, False)

def train(epoch):
    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data, target = Variable(data.cuda()), Variable(target.cuda())

        # update discriminator
        for _ in range(disc_iters):
            z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
            optim_disc.zero_grad()
            optim_gen.zero_grad()
            fake = generator(z)
            if args.loss == 'hinge':
                disc_loss_true = nn.ReLU()(1.0 - discriminator(data)).mean()
                disc_loss_fake = nn.ReLU()(1.0 + discriminator(fake)).mean()
            elif args.loss == 'wasserstein':
                disc_loss_true = -discriminator(data).mean()
                disc_loss_fake = discriminator(fake).mean()
            else:
                disc_loss_true = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda()))
                disc_loss_fake = nn.BCEWithLogitsLoss()(discriminator(fake), Variable(torch.zeros(args.batch_size, 1).cuda()))
            disc_loss = disc_loss_true + disc_loss_fake
            disc_loss.backward()
            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

        # update generator
        optim_disc.zero_grad()
        optim_gen.zero_grad()
        if args.loss == 'hinge' or args.loss == 'wasserstein':
            gen_loss = -discriminator(generator(z)).mean()
        else:
            gen_loss = nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.ones(args.batch_size, 1).cuda()))
        gen_loss.backward()
        optim_gen.step()

        if batch_idx % 50 == 0:
            print('Iter:%d' % batch_idx, '--lossD_true:%2.5f' % disc_loss_true.item(), 'lossD_fake: %2.5f' % disc_loss_fake.item(), 'lossG:%2.5f' % gen_loss.item())
            #logger('Iter:%d' % batch_idx, '--lossD_true:%2.5f' % disc_loss_true.item(), 'lossD_fake: %2.5f' % disc_loss_fake.item(), 'lossG:%2.5f' % gen_loss.item())
            vis.add_value('lossD_true', disc_loss_true.item())
            vis.add_value('lossD_fake', gen_loss.item())
            vis.add_value('lossG', gen_loss.item())

    scheduler_d.step()
    scheduler_g.step()


def evaluate(epoch):

    samples = generator(fixed_z).cpu().data.numpy()[:64]


    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    output_dir = 'samples/%s' % exp_Name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sample_path = '%s/pic_epoch%d' % (output_dir, epoch) 
    plt.savefig(sample_path, bbox_inches='tight')
    #plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)


def evaluate_IS_FID(epoch, batch_size, Z_dim, generator):
    if args.model.find('resnet') != -1:
        z_ = utils.prepare_z_dense(batch_size, Z_dim)
    else:
        z_ = utils.prepare_z(batch_size, Z_dim)
    sample = functools.partial(utils.sample_onlyZ, G=generator, z_=z_)
    IS_mean, IS_std, FID_train, FID_test = get_inception_metrics(sample, 50000, num_splits=10)
    vis.add_value('IS', IS_mean)
    vis.add_value('FID_train', FID_train)
    vis.add_value('FID_test', FID_test)
    #print('epoch %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (epoch, IS_mean, IS_std, FID))
    logger('epoch %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID_train is %5.4f, FID_test is %5.4f,' % (epoch, IS_mean, IS_std, FID_train,FID_test))

fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
os.makedirs(args.checkpoint_dir, exist_ok=True)

for epoch in range(200):
    train(epoch)
    evaluate(epoch)
    evaluate_IS_FID(epoch, args.batch_size, Z_dim, generator)
    #torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
    #torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))

