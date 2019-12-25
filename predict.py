from util_funx import process_image, predict, load_checkpoint, map_category

import torch
import argparse


# argument object
parser = argparse.ArgumentParser()


# adding arguments
# image_path = (test_dir + '/77/image_00005.jpg')
parser.add_argument('input_image', default='flower/test/77/image_00005.jpg',
                    action='store', help='Specify image location')
parser.add_argument('checkpoint', default='checkpoint.pth',
                    help='Specify checkpoint')
parser.add_argument('--top_k', type=int, default=5,
                    help='Specify top K most likely classes')
parser.add_argument('--category_names',
                    help='Specify file for the category names')
parser.add_argument('--gpu', action='store_true',
                    help='Specify the use of gpu power over cpu')


# paarsing arguments
args = parser.parse_args()


input_image = args.input_image
checkpoint = args.checkpoint
top_k = args.top_k

# gpu = args.gpu
if args.gpu and torch.cuda.is_available():
    # device agnostic to detect gpu or cpu
    device = 'cuda'
else:
    device = 'cpu'

# category_names if args.category_names else None
if args.category_names:
    category_names = args.category_names
else:
    category_names = None


# Load train model
model = load_checkpoint(checkpoint)

# Probability class
probs, classes = predict(input_image, model, top_k, device)

print('Image: ', input_image)
print('Probabities / Classes')

map_category(category_names, classes) if category_names else None
# if category_names:
#     classes = map_category(category_names, classes)

# print names of classes
for label, prob in zip(classes, probs):
    print('{:.4f} / {} '.format(prob, label))
