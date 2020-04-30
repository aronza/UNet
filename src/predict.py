import argparse
import logging
from os import listdir
from os.path import splitext, join

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import BasicDataset
from dataset.slicer import build_slices
from unet3d.model import UNet3D


def predict_img(model, patch):
    model.eval()

    with torch.no_grad():
        output = model(patch)

        # if model contains final_activation layer for normalizing logits apply it, otherwise
        # the evaluation metric will be incorrectly computed
        if hasattr(model, 'final_activation') and model.final_activation is not None:
            probs = model.final_activation(output)

        probs = probs.squeeze()

        full_mask = probs.squeeze().cpu().numpy()

        return full_mask.argmax(0).astype(np.uint8)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', required=True,
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='directory of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT',
                        help='directory of output files', required=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(input_files):
    output_files = []

    for f in input_files:
        pathsplit = splitext(f)
        output_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))

    return output_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    in_files = listdir(args.input)
    out_files = [f'OUT_{file}' for file in in_files]

    input_channels = 1
    output_channels = 2
    net = UNet3D(in_channels=input_channels, out_channels=output_channels, testing=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    logging.info(f'Using device {device}')

    logging.info("Loading model {}".format(args.model))
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        nib_img = nib.load(join(args.input, fn))
        img = nib_img.dataobj
        result_img = np.zeros(img.shape)
        slices = build_slices(img.shape)

        data_set = [BasicDataset.pre_process(img[patch], is_label=False, device=device) for patch in slices]

        loader = DataLoader(data_set, batch_size=1)

        for idx, batch in enumerate(loader):
            logging.info(f'Batch {idx} Slice: {slices[idx]}')

            result = predict_img(model=net, patch=batch)
            result_img[slices[idx]] = result

        out_fn = out_files[i]

        mask_pixels = np.count_nonzero(result_img)
        mask_percentage = (mask_pixels * 100) / result_img.size

        logging.info(f'Result Mask percentage {mask_percentage}')

        nib.save(nib.Nifti1Image(result_img, nib_img.affine), join(args.output, out_fn))

        logging.info("Mask saved to {}".format(out_files[i]))
