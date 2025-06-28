# Superresolution using an efficient sub-pixel convolutional neural network

This example illustrates how to use the efficient sub-pixel convolution layer described in ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158) for increasing spatial resolution within your network for tasks such as superresolution.

```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--seed SEED]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --cuda                use cuda
  --mps                 enable GPU on macOS
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
```

This example trains a super-resolution network on the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/), using crops from the 200 training images, and evaluating on crops of the 100 test images. A snapshot of the model after every epoch with filename model*epoch*<epoch_number>.pth

## Example Usage:

### Train with cuda

`python3 main.py --upscale_factor 3 --batchSize 256 --testBatchSize 128 --nEpochs 500 --lr 0.001 --datasetName people --cuda`

### Train continuing from checkpoint

`python3 main.py --upscale_factor 3 --batchSize 8 --testBatchSize 175 --nEpochs 100 --lr 0.001 --datasetName cars --cuda --checkpointToContinueFrom checkpoint_epoch_1.pth`


### Super Resolve

`python3 super_resolve.py --input_image dataset/BSDS300/images/test/16077.jpg --checkpoint checkpoint_epoch_1.pth --output_filename out.png --cuda`

`python3 super_resolve.py --input_image ./nc_lr --checkpoint checkpoint_epoch_1.pth --output_filename out_big.png --cuda`

### Compare models performance

`python.exe .\test_models.py --datasetname dogs --upscale_factor 3 --test_dir .\testSets\dogs_test\ --checkpoint .\checkpoints\upscale_factor_x3\dogs\checkpoint_epoch_500.pth --checkpoint_bsd .\checkpoints\upscale_factor_x3\bsd300\checkpoint_epoch_500.pth --cuda`
