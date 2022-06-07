from ..utils import argument_defaults
from . import base
from . import sc2
from . import weather
from . import traffic


def make_net(net_args):
    class_name = net_args['net_type']
    kwargs = {k: v for k, v in net_args.items() if k != 'net_type'}
    # Get desired class
    class_ = base._net_map[class_name]
    # Get default arguments for that class
    params = argument_defaults(class_.__init__)
    # Override arguments w/ provided, create
    params.update(kwargs)
    net = class_(**params)
    # Save the exact parameters we used to make this network for later
    net.args = params
    return net


def make_net_args(class_name, args):
    class_ = base._net_map[class_name]
    return class_(**args)
