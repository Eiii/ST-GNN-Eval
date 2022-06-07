from . import problem
from . import experiment
from .measure import Measure

import pickle
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as schedule
from torch.utils.data import DataLoader

from pathlib import Path
from time import time


class Trainer:
    """Defines and tracks the state of a training run"""
    def __init__(self, name, net, problem, out_path, epochs,
                 report_every=0.1, valid_every=1,
                 optim='adam', sched='cos',
                 batch_size=4,
                 lr=1e-3, min_lr=0,
                 weight_decay=0, momentum=0.95,
                 period=100,
                 num_workers=0,
                 clip_grad=None,
                 disable_valid=False,
                 reset_net=None):
        # 'Macro' parameters
        self.net = net
        self.problem = problem
        self.out_path = out_path
        self.disable_valid = disable_valid
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.clip_grad = clip_grad
        self.reset_net = reset_net
        # UI parameters
        self.report_every = report_every
        self.valid_every = valid_every
        # Set up tools
        self.measure = Measure(name)
        self.init_optim(optim, lr, weight_decay, momentum)
        self.init_sched(sched, period, lr, min_lr, epochs)

    def init_optim(self, optim_, lr, wd, mom):
        if optim_ == 'adam':
            self.optim = optim.AdamW(self.net.parameters(), lr=lr,
                                     weight_decay=wd)
        elif optim_ == 'sgd':
            self.optim = optim.SGD(self.net.parameters(), lr=lr,
                                   momentum=mom, nesterov=True,
                                   weight_decay=wd)

    def init_sched(self, sched, period, init_lr, min_lr, lr_max_epochs):
        if sched == 'cos':
            self.sched = schedule.CosineAnnealingWarmRestarts(self.optim,
                                                              period, 2,
                                                              eta_min=min_lr)
            self.batch_sched_step = lambda x: self.sched.step(100*x)
            self.epoch_sched_step = lambda: None
        elif sched == 'none':
            self.batch_sched_step = lambda x: None
            self.epoch_sched_step = lambda: None
        elif sched == 'lrtest':
            # hack in the desired values
            min_lr = min_lr
            max_lr = init_lr
            def fn(time):
                mix = time/lr_max_epochs
                goal = (min_lr**(1-mix))*(max_lr**mix)
                mult = goal/init_lr
                return mult
            self.sched = schedule.LambdaLR(self.optim, fn)
            self.batch_sched_step = lambda x: self.sched.step(x)
            self.epoch_sched_step = lambda: None
        elif sched == 'wdtest':
            # hack in the desired values
            def set_wd(time):
                wd = 1.0 - (time / lr_max_epochs)
                for g in self.optim.param_groups:
                    g['weight_decay'] = wd
            set_wd(0)
            self.batch_sched_step = lambda x: set_wd(x)
            self.epoch_sched_step = lambda: None

    def train(self):
        # Set up loaders for training&valid data
        loader_args = {'shuffle': True, 'batch_size': self.batch_size,
                       'drop_last': True, 'num_workers': self.num_workers,
                       'pin_memory': True,
                       'collate_fn': self.problem.collate_fn}
        loader = DataLoader(self.problem.train_dataset, **loader_args)
        valid_loader = DataLoader(self.problem.valid_dataset, **loader_args)
        # Useful values to use during training
        num_batches = len(loader)
        batches_elapsed = 0
        running_loss = 0
        next_train_report = self.report_every
        next_valid = self.valid_every
        # Training loop
        self.start_time = time()
        self.net = self.net.cuda()
        end_training = False
        epoch = 0
        # Eval on validation, record results
        self.validation_report(valid_loader, epoch)
        if self.reset_net:
            self.create_checkpoint()
            self.load_checkpoint()
        while not end_training:
            # Train on batch
            for i, data in enumerate(loader):
                # Check reporting & termination conditions
                batch_time = epoch + i/num_batches
                if batch_time > self.epochs:
                    end_training = True
                    break
                if batch_time > next_train_report:
                    avg_loss = running_loss / batches_elapsed
                    wall_time = self.runtime()
                    lr = self.optim.param_groups[0]['lr']
                    self.measure.training_loss(batch_time, wall_time, avg_loss, lr)
                    running_loss = 0
                    batches_elapsed = 0
                    next_train_report += self.report_every
                    if self.reset_net:
                        self.load_checkpoint()
                if batch_time > next_valid:
                    self.validation_report(valid_loader, batch_time)
                    next_valid += self.valid_every
                # Train on batch
                self.optim.zero_grad()
                data = seq_to_cuda(data)
                # Get net input from data entry & predict
                pred = self.net.forward(*self.net.get_args(data))
                # Calculate problem loss and (optionally) net loss
                loss = self.problem.loss(data, pred)
                if hasattr(self.net, 'loss'):
                    net_loss = self.net.loss(data, pred)
                    loss += net_loss
                # Optimization step
                loss.backward()
                # Clip gradients
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)
                self.optim.step()
                # Batch-wise schedule update
                self.batch_sched_step(batch_time)
                # UI reporting
                running_loss += loss.item()
                batches_elapsed += 1
            else:
                epoch += 1
                batch_time = epoch
            # Epoch-wise schedule update
            self.epoch_sched_step()
        # Final validation
        self.validation_report(valid_loader, batch_time)

    def _debug_grad_norm(self):
        norm_type = 2
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in self.net.parameters()]), norm_type)
        return total_norm

    def create_checkpoint(self):
        if self.reset_net is True:
            self.checkpoint_state = self.net.state_dict()
        elif type(self.reset_net) == str:
            reset_path = Path(self.reset_net)
            with open(reset_path, 'rb') as fd:
                data = pickle.load(fd)
                self.checkpoint_state = data['state_dict']
        else:
            raise ValueError('Unknown checkpoint argument')

    def load_checkpoint(self):
        self.net.load_state_dict(self.checkpoint_state)

    def validation_report(self, ds, epoch):
        if self.problem.valid_dataset is not None and not self.disable_valid:
            valid_loss = self.validation_loss(ds)
            wall_time = self.runtime()
            self.measure.valid_stats(epoch, wall_time, valid_loss)
        self.dump_results(epoch)

    def validation_loss(self, loader):
        self.net.eval()
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(loader):
                data = seq_to_cuda(data)
                net_args = self.net.get_args(data)
                pred = self.net(*net_args)
                loss = self.problem.loss(data, pred)
                total_loss += loss.item()
        loss = total_loss/len(loader)
        self.net.train()
        return loss

    def dump_results(self, time=None):
        state_dict = self.net.cpu().state_dict()
        data = {'measure': self.measure,
                'net_type': type(self.net).__name__,
                'net_args': self.net.args,
                'state_dict': state_dict,
                'batch_time': time}
        with self.out_path.open('wb') as fd:
            pickle.dump(data, fd)
        self.net.cuda()

    def runtime(self):
        t = time()
        if self.start_time is None:
            self.start_time = t
        return t - self.start_time


# TODO: surely this means I'm doing something wrong
def seq_to_cuda(d):
    def to_cuda(x):
        return x.cuda() if x is not None else None
    if isinstance(d, dict):
        return {k: to_cuda(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [to_cuda(v) for v in d]


def train_single(name,
                 net,
                 problem_args,
                 train_args,
                 epochs,
                 out_dir,
                 uid):
    """Main 'entry point' to train a specified network on a specified problem
    """
    prob = problem.make_problem(problem_args)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = experiment.output_path(out_dir, name, uid)
    print(out_path)
    trainer = Trainer(name, net, prob, out_path, **train_args)
    trainer.train()
    trainer.dump_results()
