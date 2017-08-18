# e-Lab Model Visualization Script
#
# Abhishek Chaurasia

import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
from graphviz import Digraph
from argparse import ArgumentParser


parser = ArgumentParser(description='e-Lab Model Visualization Script')
_ = parser.add_argument
_('--model',    type=str, default='alexnet', help='model definition')
_('--from_zoo', action='store_true', help='load from vision or your own model')
_('--detailed', action='store_true', help='detailed blocks or not')

args = parser.parse_args()


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"), format='svg')
    seen = set()

    module_att = ('kernel_size', 'stride', 'padding', 'dilation')
    detailed = args.detailed        # Select False to see concise version


    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'


    def get_details(module):
        fill_color = None
        label = str(type(module).__name__)[:-8]

        if detailed:                # Show all kernel, stride, etc. values
            if(str(type(module).__name__)=="ConvNdBackward"):
                kernel = module.next_functions[1][0].variable
                label = label + \
                        '\\nkernel_size=' + size_to_str(kernel.size())
                fill_color = 'orange'

            for attribute in module_att:
                if(hasattr(module, attribute)):
                    label = label + '\\n' + attribute + \
                            '=' + str(getattr(module, attribute))

                    if fill_color == None:
                        fill_color = 'lightblue'

            if(str(type(module).__name__)=="AddmmBackward"):  # Linear layer
                label = label + ' ' + size_to_str(module.saved_tensors[1].size())
                fill_color = 'orange'

        return label, fill_color


    def graph_gen(module):
        if module not in seen:
            seen.add(module)
            label, fill_color = get_details(module)
            dot.node(str(id(module)), label, fillcolor=fill_color)
            if hasattr(module, 'next_functions'):              # only the main branch of graph has next_function
                for child in module.next_functions:
                    if child[0]:                               # ignore variables
                        if not hasattr(child[0], 'variable'):  # eliminate accumulated grad
                            if not (str(type(child[0]).__name__)=="TransposeBackward"):
                                dot.edge(str(id(child[0])), str(id(module)))
                                graph_gen(child[0])


    graph_gen(var.grad_fn)
    return dot


if args.from_zoo:
    model = getattr(models, args.model)()
else:
    from model_def import ModelDef
    model = ModelDef()

x = torch.randn(1,3,224, 224)
y = model(Variable(x))

g = make_dot(y)
g.view()
