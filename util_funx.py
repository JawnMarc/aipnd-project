import matplotlib.pyplot as plt

import numpy as np
import torch

from torch import nn

from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models

from collections import OrderedDict
import json

from PIL import Image

import argparse


#
#
#
#
###---  Train utility functions  ---###


def process_data(dir_path):
    '''
    Arguments: the path to image data
    Returns: The loaders for train, validation and test datasets

    This function receives the directory path of the image data and apply necessary transformations

    '''

    train_dir = dir_path + '/train'
    valid_dir = dir_path + '/valid'
    test_dir = dir_path + '/test'


    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(100),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(
        train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_datasets, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(
        valid_datasets, batch_size=32, shuffle=True)


    print('<---- Loading {} directory into the network ---->'.format(dir_path))

    return trainloader, testloader, validloader, train_datasets


def model_setup(arch, hidden_units, learning_rate, output_size=120):
    '''
    Arguments: The architecture for the network(vgg16, vgg19, densenet121, resnet101), the hyperparameters for the network
    (hidden layer units, and learning rate)

    Returns: The model, criterion and optimizer for training
    '''

    if arch.lower():
        print('<---- Using the {} model arhitecture ---->'.format(arch))

    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch.lower() == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch.lower() == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch.lower() == 'resnet101':
        model = models.resnet101(pretrained=True)
    else:
        print('{} Invalid model, available models to try: vgg16, vgg19, densenet121 and resnet101'.format(arch))


    # retrieving in_features from classifier
    input_features = model.classifier[0].in_features

    # freeze parameters and creating the classifier
    for param in model.parameters():
        param.requires_grad = False

    # defining the classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout()),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, classifier, criterion, optimizer


# a function for test loss and validation pass
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        # forward pass
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # calculating accuracy
        # model return is log-softmax, take exponential to get the probabilities
        # class with hihgest prob. iis our predicted class, compare to true label
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])

        # accuracy is the number of correct predictions divided by all prediction (take the mean)
        accuracy += equality.type(torch.cuda.FloatTensor).mean()

    return test_loss, accuracy


# function to training the model

def train_neural(model, trainloader, validloader, criterion, optimizer, device, epochs=10, print_every=40):
    '''
    Arguments: The model, dataset of trainloader and validloader, criterion, the optimizer, choice of gpu power or cpu, the number of epochs,

    Returns: Nothing

    This function trains the model over a certain number of epochs and displays the training,validation and accuracy every "print_every" step using cuda if specified. The training method is specified by the criterion and the optimizer which are NLLLoss and Adam respectively
    '''

    print('<---- Starting Neural Network Training ---->')
    steps = 0
    running_loss = 0
    model.to(device)

    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            # setting gradients to zero
            optimizer.zero_grad()

            # forward and backward pass
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()

            # updating weights
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # turns out dropout mode in inference
                model.eval()

                # turns off gradient for validation, saves memory and computation
                with torch.no_grad():
                    valid_loss, accuracy = validation(
                        model, validloader, criterion, device)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(
                          valid_loss/len(validloader)),
                      "Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()


def save_checkpoint(model, arch, save_dir, classifier, optimizer, train_datasets, epochs=10):
    '''
    Arguments: The saving path and the hyperparameters of the network

    Returns: Nothing

    This function saves the model at a specified by the user path
    '''

    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {
        'model': model,
        'arch': arch,
        'classifier': classifier,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs
    }

    path = save_dir + 'checkpoint.pth'
    torch.save(checkpoint, path)

    print('<---- Checkpoint saved to: {}---->'.format(path))


#
#
#
#
###---  Predict utility functions  ---###


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint['optimizer']
    model.epochs = checkpoint['epochs']
#     model.criterion = checkpoint['criterion']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    # scale, crop and normalize pil image as manner as trained using transforms.Compose()
    pil_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # open image and apply pil_transforms
    pil_image = Image.open(image)
    pil_image = pil_transform(pil_image)

    return pil_image



def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

#     model.to('cpu')
    model.eval()

    if device == 'cuda':
        model.to('cuda')
    else:
        model.to('cpu')

    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.float()

    if device == 'cuda':
         with torch.no_grad():
            output = model.forward(image.cuda())
    else:
        model.to('cpu')
        with torch.no_grad():
            output = model.forward(image)


    probability = torch.exp(output)

    # highest k prob tensors and their indices
    probs, indices = probability.topk(topk)

    probs = probs.cpu()
    indices = indices.cpu()

    # dict inverse
    invert_map = {index: itm for itm, index in model.class_to_idx.items()}

    classes = []
    for index in indices.numpy()[0]:
        classes.append(invert_map[index])



    return probs.numpy()[0], classes



def map_category(file, classes):
    class_list = []

    with open(file, 'r') as f:
        cat_to_names = json.load(f)

    for cls in classes:
        class_list.append(cat_to_names[cls])

    return class_list

