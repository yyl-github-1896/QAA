#!/bin/bash
source='./adv_imgs/apot/resnet56_120603_w2a2_stochastic/pgd/'

echo "-----------------------------------------------"
python evaluate.py --arch "resnet18" --quantize_method "standard_fp" --w_bit 32 --a_bit 32 --output_dir $source
python evaluate.py --arch "resnet50" --quantize_method "standard_fp" --w_bit 32 --a_bit 32 --output_dir $source
python evaluate.py --arch "vgg19_bn" --quantize_method "standard_fp" --w_bit 32 --a_bit 32 --output_dir $source
python evaluate.py --arch "densenet121" --quantize_method "standard_fp" --w_bit 32 --a_bit 32 --output_dir $source
python evaluate.py --arch "mobilenet_v2" --quantize_method "standard_fp" --w_bit 32 --a_bit 32 --output_dir $source