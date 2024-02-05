#!/bin/bash
source="./adv_imgs/qdrop/vgg16_w2a2_stochastic/ssa"


echo "======================================================="
python evaluate.py --arch "inception_v3" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
echo "-------------------------------------------------------"
python evaluate.py --arch "inception_v4" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
echo "-------------------------------------------------------"
python evaluate.py --arch "inception_resnet_v2" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
echo "-------------------------------------------------------"
python evaluate.py --arch "resnet50" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
echo "-------------------------------------------------------"
python evaluate.py --arch "resnet152" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source --batch_size 2
echo "-------------------------------------------------------"
python evaluate.py --arch "vgg16" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
echo "-------------------------------------------------------"
python evaluate.py --arch "vgg19" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "-------------------------------------------------------"
# python evaluate.py --arch "mobilenet_v2" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "-------------------------------------------------------"
# python evaluate.py --arch "squeezenet1_1" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "-------------------------------------------------------"
# python evaluate.py --arch "shufflenet_v2_x1_0" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "-------------------------------------------------------"
# python evaluate.py --arch "efficientnet_b0" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "-------------------------------------------------------"
# python evaluate.py --arch "mobilevitv2-0.5" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "======================================================="



# python evaluate.py --arch "pit" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "-------------------------------------------------------"
# python evaluate.py --arch "cait" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "-------------------------------------------------------"
# python evaluate.py --arch "deit" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
# echo "-------------------------------------------------------"
# python evaluate.py --arch "swin" --quantize_method "fp" --w_bit 32 --a_bit 32 --output_dir $source
