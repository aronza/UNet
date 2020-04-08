import torch
import logging
from torch import nn

from unet3d.utils import RunningAverage
from unet3d.metrics import BoundaryAdaptedRandError


# def eval_net(net, loader, device, n_val):
#     """Evaluation without the densecrf with the dice coefficient"""
#     net.eval()
#     tot = 0
#
#     with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
#         for batch in loader:
#             imgs = batch['image']
#             true_masks = batch['mask']
#
#             imgs = imgs.to(device=device, dtype=torch.float32)
#             mask_type = torch.float32 if net.n_classes == 1 else torch.long
#             true_masks = true_masks.to(device=device, dtype=mask_type)
#
#             mask_pred = net(imgs)
#
#             for true_mask, pred in zip(true_masks, mask_pred):
#                 pred = (pred > 0.5).float()
#                 if net.n_classes > 1:
#                     tot += F.cross_entropy(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0)).item()
#                 else:
#                     tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
#             pbar.update(imgs.shape[0])
#
#     return tot / n_val


def validate(model, val_loader, device, loss_fnc=nn.BCEWithLogitsLoss, eval_criterion=BoundaryAdaptedRandError()):
    logging.info('Validating...')

    val_losses = RunningAverage()
    val_scores = RunningAverage()

    with torch.no_grad():
        for i, t in enumerate(val_loader):
            logging.info(f'Validation iteration {i}')

            input, target = _split_training_batch(t, device)

            # forward pass
            output = model(input)

            # compute the loss
            loss = loss_fnc(output, target)

            val_losses.update(loss.item(), _batch_size(input))

            # if model contains final_activation layer for normalizing logits apply it, otherwise
            # the evaluation metric will be incorrectly computed
            if hasattr(model, 'final_activation') and model.final_activation is not None:
                output = model.final_activation(output)

            eval_score = eval_criterion(output, target)
            val_scores.update(eval_score.item(), _batch_size(input))

        logging.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
        return val_scores.avg


def _split_training_batch(t, device):
    def _move_to_device(input):
        if isinstance(input, tuple) or isinstance(input, list):
            return tuple([_move_to_device(x) for x in input])
        else:
            return input.to(device)

    t = _move_to_device(t)
    input, target = t

    return input, target


@staticmethod
def _batch_size(input):
    if isinstance(input, list) or isinstance(input, tuple):
        return input[0].size(0)
    else:
        return input.size(0)
