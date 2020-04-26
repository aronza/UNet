import logging

import torch

from unet3d.utils import RunningAverage


def validate(model, val_loader, loss_fnc, eval_criterion):
    logging.info('\nValidating...')

    val_losses = RunningAverage()
    val_scores = RunningAverage()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            logging.info(f'Validation iteration {i}')

            img = batch['image']
            mask = batch['mask']

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
            logging.info(f'I: {i}, Validation Score: {eval_score}')

            val_scores.update(eval_score.item(), img.shape[0])

        logging.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
        return val_scores.avg
