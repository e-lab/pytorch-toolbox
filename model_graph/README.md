# Generate Model Graph

This script uses the backward pass of a model to generate the model graph and displays it using graphviz.
You can load model definition from torchvision or your own model definition.
In order to load your own model definition, modify `model_def.py` file.

+ From torchvision:

```
python3 visualize.py --from_zoo --model resent18 --detailed
```

+ Custom model definition:

```
python3 visualize.py --detailed
```

Adapted from: https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py

Also see: https://discuss.pytorch.org/t/print-autograd-graph/692
