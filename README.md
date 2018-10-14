# universal_transformer

## How to use
Use [run.sh](run.sh) to train.

The argument `--t2t_usr_dir=.` registers the model into tensor2tensor.

Please refer to tensor2tensor documentation for other usages.

## Dependency
#### tensor2tensor
Installation:
```
# Assumes tensorflow or tensorflow-gpu installed
pip install tensor2tensor

# Installs with tensorflow-gpu requirement
pip install tensor2tensor[tensorflow_gpu]

# Installs with tensorflow (cpu) requirement
pip install tensor2tensor[tensorflow]
```

## References
Implementation of transformer:
[tensor2tensor/models](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models)

Implementation of Grave's ACT for RNN:
[DeNeutoy/act-tensorflow](https://github.com/DeNeutoy/act-tensorflow)
