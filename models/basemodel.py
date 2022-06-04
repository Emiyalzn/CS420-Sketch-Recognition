import os
import torch

class BaseModel(object):
    def __init__(self):
        self._nets = list()
        self._net_names = list()
        self._train_flags = list()

    def __call__(self):
        pass

    def register_nets(self, nets, names, train_flags):
        self._nets.extend(nets)
        self._net_names.extend(names)
        self._train_flags.extend(train_flags)

    def params(self, trainable, named=False, add_prefix=False):
        def _get_net_params(_net, _net_name):
            if named:
                if add_prefix:
                    return [
                        (_net_name + '.' + _param_name, _param_data) for _param_name, _param_data in
                        _net.named_parameters()
                    ]
                else:
                    return list(_net.named_parameters())
            else:
                return list(_net.parameters())

        res = list()
        for idx, net in enumerate(self._nets):
            net_flag = self._train_flags[idx]
            net_name = self._net_names[idx]

            if trainable:
                if net_flag:
                    res.extend(_get_net_params(net, net_name))
            else:
                res.extend(_get_net_params(net, net_name))
        return res

    def print_params(self):
        print('[*] Model Parameters:')
        for nid, net in enumerate(self._nets):
            if self._train_flags[nid]:
                print('[*]  Trainable Module {}'.format(self._net_names[nid]))
            else:
                print('[*]  None-Trainable Module {}'.format(self._net_names[nid]))
            for name, param in net.named_parameters():
                print('[*]    {}: {}'.format(name, param.size()))
        print('[*] Model Size: {:.5f}M'.format(self.num_params() / 1e6))

    def subnet_dict(self):
        return {self._net_names[i]: self._nets[i] for i in range(len(self._nets))}

    def save(self, root_folder, iterations, prefix=''):
        path_prefix = os.path.join(root_folder, prefix + '_iter_{}'.format(iterations))
        res = list()
        for net, name in zip(self._nets, self._net_names):
            net_path = path_prefix + '_' + name + '.pth'
            torch.save(net.state_dict(), net_path)
            res.append(net_path)
        return res

    def load(self, path_prefix, net_names=None, strict=True):
        res = list()
        if net_names is None or len(net_names) == 0:
            for net, name in zip(self._nets, self._net_names):
                net_path = path_prefix + '_' + name + '.pth'
                if os.path.exists(net_path):
                    net.load_state_dict(torch.load(net_path), strict=strict)
                    res.append(net_path)
        else:
            for net, name in zip(self._nets, self._net_names):
                if name not in net_names:
                    continue
                net_path = path_prefix + '_' + name + '.pth'
                if not os.path.exists(net_path):
                    raise Exception("[!] {} does not exist.".format(net_path))
                net.load_state_dict(torch.load(net_path), strict=strict)
                res.append(net_path)
        return res

    def train_mode(self):
        for net, train_flag in zip(self._nets, self._train_flags):
            if train_flag:
                net.train()
            else:
                net.eval()

    def eval_mode(self):
        for net in self._nets:
            net.eval()

    def to(self, device):
        for net in self._nets:
            net.to(device)