import torch


class SelfEnsemble:
    def __init__(self, net, count, kw={}):
        self.net = net
        self.count = count
        self.kw = kw

    def get_args(self, item):
        return self.net.get_args(item)

    def __call__(self, *args):
        preds = [self.net(*args, **self.kw) for _ in range(self.count)]
        return torch.stack(preds).mean(dim=0)
