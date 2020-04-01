import os
import torch
from tensorboardX import SummaryWriter

from shutil import copy, rmtree


class Logger:
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.logfile = open(os.path.join(logdir, 'log.txt'), 'w')
        train_writer_dir = os.path.join(logdir, 'train')
        eval_writer_dir = os.path.join(logdir, 'eval')
        # Remove old tf events
        if os.path.exists(train_writer_dir):
            rmtree(train_writer_dir)
        if os.path.exists(eval_writer_dir):
            rmtree(eval_writer_dir)
        self.train_writer = SummaryWriter(os.path.join(logdir, 'train'))
        self.eval_writer = SummaryWriter(os.path.join(logdir, 'eval'))

    def log_string(self, out_str, do_print=True):
        self.logfile.write(str(out_str) + '\n')
        self.logfile.flush()
        if do_print:
            print(out_str)

    def backup_files(self, file_list):
        for filepath in file_list:
            copy(filepath, self.logdir)

    def close(self):
        self.logfile.close()
        self.train_writer.close()
        self.eval_writer.close()

    def log_scalar_train(self, tag, value, global_step):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.train_writer.add_scalar(tag, value, global_step)

    def log_scalar_eval(self, tag, value, global_step):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.eval_writer.add_scalar(tag, value, global_step)
