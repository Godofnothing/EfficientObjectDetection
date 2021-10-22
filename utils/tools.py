import numpy as np
import torch
import matplotlib.pyplot as plt

class StatManagerBase:
    
    @staticmethod
    def _batch_mean(inst, value):
        inst.curr_epoch_history.append(value.mean().item())
    
    @staticmethod
    def _batch_extend(inst, value):
        inst.curr_epoch_history.extend(value.detach().cpu().tolist())
    
    @staticmethod
    def _batch_std(inst, value):
        inst.curr_epoch_history.append(value.std().item())

class StatManager(StatManagerBase):

    reduction_funcs = {
        'batch_mean': StatManagerBase._batch_mean,
        'batch_extend': StatManagerBase._batch_extend,
        'batch_std': StatManagerBase._batch_std
    }
    
    def do_reduction(self, command):
        assert command in ['update', 'new_epoch']
        
        def on_update(value):
            functions = self.reduction_funcs[self.reduction]
            if isinstance(functions, dict):
                func = functions['update']
            else:
                func = functions
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            func(self, value)
        
        def on_new_epoch():
            functions = self.reduction_funcs[self.reduction]
            if isinstance(functions, dict):
                functions['new_epoch'](self)
        
        if command == 'update':
            return on_update
        return on_new_epoch
    
    def batch_extend_reduction(self, value):
        self.curr_epoch_history.extend(value.detach().cpu().tolist())
    
    def __init__(self, name, reduction='batch_mean'):
        '''
        This class helps to collect statistics during the training
        '''
        self.name = name
        self.history = []
        self.curr_epoch_history = []
        assert reduction in self.reduction_funcs.keys()
        self.reduction = reduction
        self.curr_epoch = 0
    
    def reset(self):
        self.history = []
        self.curr_epoch_history = []
        self.curr_epoch = 0
    
    def add(self, value):
        self.do_reduction('update')(value)

    def epoch(self):
        self.curr_epoch += 1
        self.do_reduction('new_epoch')()
        self.history.append(self.curr_epoch_history)
        self.curr_epoch_history = []
    
    def get_linearized_history(self, only_last_epoch=False):
        lin_history = []
        if not only_last_epoch:
            lin_history = [item for sublist in self.history for item in sublist]
        lin_history.extend(self.curr_epoch_history)
        return lin_history

    def draw(self, ax, draw_last_epoch_policy='standard', only_last_epoch=True):
        assert draw_last_epoch_policy in ['standard', 'adaptive']
        lh = len(self.history)
        if lh > 0 and not only_last_epoch:
            xs = np.concatenate([
                np.linspace(i, i + 1, len(self.history[i]), endpoint=False) \
                for i in range(lh)])
        else:
            xs = np.array([])
        if len(self.curr_epoch_history) > 0:
            if draw_last_epoch_policy == 'standard':
                end_ls = lh + 1
            else:
                if lh > 0 and not only_last_epoch:
                    mean_len_histories = int(np.mean([len(hist) for hist in self.history]))
                else:
                    mean_len_histories = len(self.curr_epoch_history)
                end_ls = lh + len(self.curr_epoch_history)/float(mean_len_histories)
            xs = np.concatenate([
                xs, np.linspace(
                    lh, end_ls, len(self.curr_epoch_history), endpoint=False)])
        if len(xs) > 0:
            linearized_history = self.get_linearized_history(
                only_last_epoch=only_last_epoch)
            ax.plot(xs, linearized_history, label=self.name)

class StatsSuiteManager:
    
    def __init__(self):
        '''
        This class stores and draws several statistics
        '''
        self.stats_managers = {}
        self.set_naxes = set()
    
    def register(self, stat_manager:StatManager, n_ax):
        assert isinstance(stat_manager, StatManager)
        assert isinstance(n_ax, int)
        self.stats_managers[stat_manager.name] = [stat_manager, n_ax]
        self.set_naxes.add(n_ax)
    
    def add(self, name, value):
        self.stats_managers[name][0].add(value)
    
    def epoch(self, *names):
        if len(names) == 0:
            for value in self.stats_managers.values():
                value[0].epoch()
            return
        for name in names:
            self.stats_managers[name][0].epoch()
    
    def draw(self, *args, ncols=1, figsize=(10, 10)):
        for n_ax in args:
            assert isinstance(n_ax, int)
            assert n_ax in self.set_naxes
        if len(args) == 0:
            args = list(self.set_naxes)
        len_axs = len(args)
        nrows = int(np.ceil(len_axs/float(ncols)))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if isinstance(axs, np.ndarray):
            axs = axs.flatten()
        else:
            axs = np.asarray([axs,])
        axs_dict = {args[i]: axs[i] for i in range(len_axs)}
        for value in self.stats_managers.values():
            if value[1] in args:
                ax = axs_dict[value[1]]
                value[0].draw(ax)
                ax.legend()
        # plt.legend()
        plt.tight_layout()
        plt.show()
