

from util_funx import process_data, model_setup, train_neural, save_checkpoint

import torch
import argparse

# argument object
parser = argparse.ArgumentParser()


# adding arguments
parser.add_argument('data_dir', default='./flowers/', action='store',
                    help='Specify the image data directory')
parser.add_argument('--save_dir', default='./',
                    help='Specify directory to save file')
parser.add_argument('--arch', default='vgg16',
                    help='Specify the model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='Specify the laerning rate for your model')
parser.add_argument('--hidden_layers', type=int, default=512,
                    help='Specify the hidden units of your model')
parser.add_argument('--epochs', type=int, default=10,
                    help='Specify the number of epochs')
parser.add_argument('--gpu', action='store', default='gpu',
                    help='Specify the use of gpu power over cpu')

# paarsing arguments
args = parser.parse_args()

dir_path = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
architect = args.arch
hidden_layer = args.hidden_layers
epochs = args.epochs
gpu = args.gpu

# device agnostic to detect gpu or cpu
if gpu and torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load datasets and apply transformations
trainloader, testloader, validloader, train_datasets = process_data(dir_path)

# Model setup
model, classifier, criterion, optimizer = model_setup(
    architect, hidden_layer, lr)

# Network trains
train_neural(model, trainloader, validloader,
             criterion, optimizer, device, epochs)

# Saved trained network
save_checkpoint(model, architect, save_dir,
                classifier, optimizer, train_datasets)
