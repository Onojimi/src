import os
import argparse
import numpy as np
from PIL import Image

import torch

from unet.unet_model import Unet
from utils.load import get_ids, split_ids, get_imgs_and_masks
from utils.utils import split_train_val, batch, normalize, resize_and_crop,\
    hwc_to_chw

from torchvision import transforms

def predict_img(net,
                full_img,
                scale_factor = 1,
                out_threshold = 0.5,
                use_dense_crf = False,
                use_gpu = True):
    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]
    
    img = resize_and_crop(full_img,scale = scale_factor)
    img = normalize(img)
    img = hwc_to_chw(img)
    img.torch.from_numpy()
    
    if use_gpu:
        img = img.cuda()
    
    with torch.no_grad():
        img_probs = net(img)
        img_probs = img_probs.squeeze(0)
        
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(img_height),
                transforms.ToTensor()
            ]
        )
        
        img_probs = tf(img)
        img_mask_np = img_probs.squeeze().cpu().numpy()
    
#     if use_dense_crf:
#         img_mask = dense_crf(np.array(full_img))
    return img_mask_np > out_threshold
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='model_fin.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'model_fin.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()     

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files   

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = Unet(n_channels=3, n_classes=1)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")
    
    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

#         if args.viz:
#             print("Visualizing results for image {}, close to continue ...".format(fn))
#             plot_img_and_mask(img, mask)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            print("Mask saved to {}".format(out_files[i]))
    