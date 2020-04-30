import argparse
import logging
import os
import sys
import timeit
from datetime import datetime

import matplotlib as mpl
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

mpl.use('Agg')

import matplotlib.pyplot as plt

from dataset.dataset import BasicDataset
from eval import validate
from unet3d.metrics import MeanIoU
from unet3d.model import UNet3D

dir_img = '/data/h_oguz_lab/larsonke/Raw/Training-Data/T1/'
dir_mask = '/data/h_oguz_lab/larsonke/Raw/Training-Data/Brainmask/'
dir_checkpoint = 'checkpoints/'


def plot_cost(costs, name, model_name):
    plt.clf()
    plt.title(name + " over Gradient Descent Iterations")
    plt.xlabel("Iterations")
    plt.ylabel(name)
    plt.plot(range(len(costs)), costs, color='red')  # cost line
    plt.savefig('runs/' + model_name + str(datetime.now()) + '_' + name + '_overtime.png')


def train_net(model: UNet3D,
              device,
              loss_fnc=CrossEntropyLoss(),
              eval_criterion=MeanIoU(),
              epochs=5,
              batch_size=1,
              learning_rate=0.0002,
              val_percent=0.01,
              test_percent=0.1,
              name='U-Net',
              save_cp=True):
    data_set = BasicDataset(dir_img, dir_mask, 'T1', device)
    train_loader, val_loader, test_loader = data_set.split_to_loaders(val_percent, test_percent, batch_size)

    writer = SummaryWriter(comment=f'LR_{learning_rate}_BS_{batch_size}')
    global_step = 0
    logging.info(f'''Starting {name} training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_loader)}
        Validation size: {len(val_loader)}
        Testing size:    {len(test_loader)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    losses = []
    val_scores = []

    for epoch in range(epochs):

        epoch_loss = 0
        for batch in train_loader:
            model.train()
            start_time = timeit.default_timer()

            img = batch['image']
            mask = batch['mask']

            masks_pred = model(img)

            loss = loss_fnc(masks_pred, mask)

            epoch_loss += loss.item()
            losses.append(loss.item())

            writer.add_scalar('Loss/train', loss.item(), global_step)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            global_step += 1
            elapsed = timeit.default_timer() - start_time
            logging.info(f'I: {global_step}, Loss: {loss.item()} in {elapsed} seconds')

            if global_step % (len(train_loader) // (5 * batch_size)) == 0:
                val_score = validate(model, val_loader, loss_fnc, eval_criterion)
                val_scores.append(val_score)

                writer.add_scalar('Validation/test', val_score, global_step)

        if save_cp:
            plot_cost(losses, name='Loss' + str(epoch), model_name=name)
            plot_cost(val_scores, name='Validation' + str(epoch), model_name=name)

            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       dir_checkpoint + f'{name}_epoch{epoch + 1}.pth')
            logging.info(f'Epoch: {epoch + 1} Loss: {epoch_loss}')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0002,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-n', '--name', dest='name', type=str, default='U-Net',
                        help='Prefix name to be used in output files')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=1.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    input_channels = 1
    output_channels = 2
    net = UNet3D(in_channels=input_channels, out_channels=output_channels)
    logging.info(f'Network:\n'
                 f'\t{input_channels} input channels\n'
                 f'\t{output_channels} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(model=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  learning_rate=args.lr,
                  device=device,
                  val_percent=args.val / 100,
                  name=args.name)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
