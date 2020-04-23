import logging

import torch

from unet3d.utils import RunningAverage


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

        
def validate(model, val_loader, loss_fnc, eval_criterion, device):
    logging.info('Validating...')

    val_losses = RunningAverage()
    val_scores = RunningAverage()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logging.info(f'Validation iteration {i}')

            img = batch['image']
            mask = batch['mask']
            
            img = img.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=torch.float32)

            # forward pass
            output = model(img)

            # compute the loss
            loss = loss_fnc(output, mask)

            val_losses.update(loss.item(), img.shape[0])

            # if model contains final_activation layer for normalizing logits apply it, otherwise
            # the evaluation metric will be incorrectly computed
            if hasattr(model, 'final_activation') and model.final_activation is not None:
                output = model.final_activation(output)

            eval_score = eval_criterion(output, mask)
            val_scores.update(eval_score.item(), img.shape[0])

        logging.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
        return val_scores.avg

