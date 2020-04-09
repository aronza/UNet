import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import validate
from unet3d.model import UNet3D

from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = '/data/h_oguz_lab/larsonke/Raw/Training-Data/T1/'
dir_mask = '/data/h_oguz_lab/larsonke/Raw/Training-Data/WM/'
dir_checkpoint = 'checkpoints/'


def train_net(model: UNet3D,
              device,
              loss_fnc,
              epochs=5,
              batch_size=1,
              learning_rate=0.0002,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    data_set = BasicDataset(dir_img, dir_mask, 'T1', 'wm', img_scale)
    n_val = int(len(data_set) * val_percent)
    n_train = len(data_set) - n_val
    train, val = random_split(data_set, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{learning_rate}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                # assert imgs.shape[1] == model.n_channels, \
                #     f'Network has been defined with {model.n_channels} input channels, ' \
                #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = model(imgs)
                loss = loss_fnc(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (len(data_set) // (10 * batch_size)) == 0:
                    val_score = validate(model, val_loader, device)

                    writer.add_scalar('Validation/test', val_score, global_step)
                    writer.add_images('images', imgs, global_step)
                    if model.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(model.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    input_channels = 1
    output_channels = 1
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
                  loss_fnc=nn.CrossEntropyLoss if output_channels > 1 else nn.BCEWithLogitsLoss,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
