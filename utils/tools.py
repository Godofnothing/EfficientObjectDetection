import numpy as np
import torch
import matplotlib.pyplot as plt
from collections.abc import Iterable
import pickle

def compute_s2(cum, cum2, n, unbiased=True):
        s2 = 0.
        n = float(n)
        if unbiased:
            if n > 1:
                s2 = cum2 / (n - 1.) - (cum**2) / (n * (n - 1.))
        else:
            s2 = cum2 / n - (cum/n) ** 2
        return s2

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

    @staticmethod
    def _epoch_stat_update(mode):
        assert mode in ['mean', 'std']

        def epoch_stat_update_func(inst, value):
            if not hasattr(inst, '_curr_epoch_n'):
                inst._curr_epoch_n = 0
                inst._curr_epoch_cum = 0.
                if mode == 'std':
                    inst._curr_epoch_cum2 = 0.
            assert len(value.shape) < 2
            if len(value.shape) == 0:
                len_value = 1
            else:
                len_value = value.size(0)
            inst._curr_epoch_cum += value.sum().item()
            inst._curr_epoch_n += len_value
            if mode == 'std':
                inst._curr_epoch_cum2 += (value**2).sum().item()

        return epoch_stat_update_func

    @staticmethod
    def _epoch_stat_new_epoch(mode):
        assert mode in ['mean', 'std']

        def epoch_stat_new_epoch_func(inst):
            if not hasattr(inst, '_curr_epoch_n'):
                inst._curr_epoch_n = 0
                inst._curr_epoch_cum = 0.
                if mode == 'std':
                    inst._curr_epoch_cum2 = 0.
            if inst._curr_epoch_n == 0:
                raise Exception(f"No updates during the epoch, statistic '{inst.name}.{mode}'!")
            if mode == 'mean':
                inst.curr_epoch_history = [
                    inst._curr_epoch_cum/float(inst._curr_epoch_n), ]
            if mode == 'std':
                inst.curr_epoch_history = [
                    np.sqrt(compute_s2(
                        inst._curr_epoch_cum,
                        inst._curr_epoch_cum2,
                        inst._curr_epoch_n)), ]
            inst._curr_epoch_n = 0
            inst._curr_epoch_cum = 0.
            if mode == 'std':
                inst._curr_epoch_cum2 = 0.

        return epoch_stat_new_epoch_func
    
    @staticmethod
    def _epoch_stat_draw_start(mode):
        assert mode in ['mean', 'std']
        
        def epoch_stat_draw_func(inst):
            if not hasattr(inst, '_curr_epoch_n'):
                inst._curr_epoch_n = 0
                inst._curr_epoch_cum = 0.
                if mode == 'std':
                    inst._curr_epoch_cum2 = 0.
            if inst._curr_epoch_n == 0:
                return None
            if mode == 'mean':
                inst.curr_epoch_history = [
                    inst._curr_epoch_cum/float(inst._curr_epoch_n), ]
            if mode == 'std':
                inst.curr_epoch_history = [
                    np.sqrt(compute_s2(
                        inst._curr_epoch_cum,
                        inst._curr_epoch_cum2,
                        inst._curr_epoch_n)), ]

        return epoch_stat_draw_func

def stat_manager_todict(sm):
    sm_dict = {
        'name': sm.name,
        'history': sm.history,
        'epoch': sm.curr_epoch,
        'reduction': sm.reduction}
    return sm_dict

def stat_manager_fromdict(sm_dict, draw_only_last_epoch=True):
    sm = StatManager(
        sm_dict['name'], 
        reduction=sm_dict['reduction'],
        draw_only_last_epoch=draw_only_last_epoch)
    sm.history = sm_dict['history']
    sm.curr_epoch = sm_dict['epoch']
    return sm

class StatManager(StatManagerBase):

    reduction_funcs = {
        'batch_mean': StatManagerBase._batch_mean,
        'batch_extend': StatManagerBase._batch_extend,
        'batch_std': StatManagerBase._batch_std,
        'epoch_mean': {
            'update': StatManagerBase._epoch_stat_update('mean'),
            'new_epoch': StatManagerBase._epoch_stat_new_epoch('mean'),
            'draw_start': StatManagerBase._epoch_stat_draw_start('mean')},
        'epoch_std': {
            'update': StatManagerBase._epoch_stat_update('std'),
            'new_epoch': StatManagerBase._epoch_stat_new_epoch('std'),
            'draw_start': StatManagerBase._epoch_stat_draw_start('std')}
    }

    def do_reduction(self, command):
        assert command in [
            'update', 'new_epoch', 'draw_start', 'draw_finish']
        functions = self.reduction_funcs[self.reduction]

        def on_update(value):
            if isinstance(functions, dict):
                func = functions['update']
            else:
                func = functions
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            func(self, value)

        def on_new_epoch():
            if isinstance(functions, dict):
                if 'new_epoch' in functions.keys():
                    functions['new_epoch'](self)

        def on_draw_start():
            if isinstance(functions, dict):
                if 'draw_start' in functions.keys():
                    functions['draw_start'](self)

        def on_draw_finish():
            if isinstance(functions, dict):
                if 'draw_finish' in functions.keys():
                    functions['draw_finish'](self)

        if command == 'update':
            return on_update
        if command == 'new_epoch':
            return on_new_epoch
        if command == 'draw_start':
            return on_draw_start
        if command == 'draw_finish':
            return on_draw_finish
        raise Exception('something has gone wrong')

    def batch_extend_reduction(self, value):
        self.curr_epoch_history.extend(value.detach().cpu().tolist())

    def __init__(self, name, reduction='batch_mean', draw_only_last_epoch=True):
        '''
        This class helps to collect statistics during the training
        '''
        self.name = name
        self.history = []
        self.curr_epoch_history = []
        assert reduction in self.reduction_funcs.keys()
        self.reduction = reduction
        self.curr_epoch = 0
        self.draw_last_epoch = draw_only_last_epoch

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

    def draw(self, ax, draw_last_epoch_policy='standard'):
        assert draw_last_epoch_policy in ['standard', 'adaptive']
        self.do_reduction('draw_start')()
        lh = len(self.history)
        if lh > 0 and not self.draw_last_epoch:
            xs = np.concatenate([
                np.linspace(i, i + 1, len(self.history[i]), endpoint=False) \
                for i in range(lh)])
        else:
            xs = np.array([])
        if len(self.curr_epoch_history) > 0:
            if draw_last_epoch_policy == 'standard':
                end_ls = lh + 1
            else:
                if lh > 0 and not self.draw_last_epoch:
                    mean_len_histories = int(np.mean([len(hist) for hist in self.history]))
                else:
                    mean_len_histories = len(self.curr_epoch_history)
                end_ls = lh + len(self.curr_epoch_history)/float(mean_len_histories)
            xs = np.concatenate([
                xs, np.linspace(
                    lh, end_ls, len(self.curr_epoch_history), endpoint=False)])
        if len(xs) > 0:
            linearized_history = self.get_linearized_history(
                only_last_epoch=self.draw_last_epoch)
            assert len(xs) == len(linearized_history)
            if len(xs) > 1:
                ax.plot(xs, linearized_history, label=self.name)
            else:
                ax.scatter(xs, linearized_history, label=self.name)
        self.do_reduction('draw_finish')()

def stats_suite_manager_todict(ssm):
    ssm_dict = {key: stat_manager_todict(
        value[0]) for key, value in ssm.stats_managers.items()}
    return ssm_dict

def stats_suite_manager_fromdict(ssm_dict, naxs=None, last_epoch_draw=True):
    if naxs is None:
        naxs = list(range(len(ssm_dict)))
    assert isinstance(naxs, (list, dict))
    if not len(naxs) == len(ssm_dict):
        raise Exception(f"len of 'naxs' = {len(naxs)} must" + \
                        " coinside with len of 'ssm_dict' = {len(ssm_dict)}")
    if isinstance(last_epoch_draw, bool):
        last_epoch_draw = [last_epoch_draw] * len(ssm_dict)
    assert isinstance(last_epoch_draw, (list, dict))
    if not len(last_epoch_draw) == len(ssm_dict):
        raise Exception(f"len of 'last_epoch_draw' = {len(last_epoch_draw)} must" + \
                        f" coinside with len of 'ssm_dict' = {len(ssm_dict)}")
    ssm = StatsSuiteManager()
    for i_stat, (stat, stat_dict) in enumerate(ssm_dict.items()):
        if isinstance(naxs, list):
            nax = naxs[i_stat]
        elif isinstance(naxs, dict):
            nax = naxs[stat]
        if isinstance(last_epoch_draw, list):
            led = last_epoch_draw[i_stat]
        elif isinstane(last_epoch_draw, dict):
            led = last_epoch_draw[stat]
        ssm.register(stat_manager_fromdict(stat_dict, led), nax)
    return ssm

def stats_suite_manager_serialize(ssm, file_name):
    ssm_dict = stats_suite_manager_todict(ssm)
    with open(file_name, 'wb') as f:
        pickle.dump(ssm_dict, f)

def stats_suite_manager_deserialize(file_name, **kwargs):
    '''
    for kwargs reference see `stats_suite_manager_fromdict`
    '''
    with open(file_name, 'rb') as f:
        ssm_dict = pickle.load(f)
    ssm = stats_suite_manager_fromdict(ssm_dict, **kwargs)
    return ssm

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
