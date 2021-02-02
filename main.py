import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str, default='dvae',help='dvae, vaeUnet, unet')


args = parser.parse_args()

if args.model == 'dvae':
    os.system('cd dnoiseVAE/src\npython3 dnoiseVAE.py')
if args.model == 'vaeUnet':
    os.system('cd lungVAE\npython3 train.py')
if args.model == 'unet':
    os.system('cd lungVAE\npython3 train.py --unet')


