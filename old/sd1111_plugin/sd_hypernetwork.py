import csv
import os
import sys
import traceback
from pathlib import Path

import src_plugins.sd1111_plugin.options
import src_plugins.sd1111_plugin.sd_paths
import src_plugins.sd1111_plugin.sd_textinv_dataset
import torch
from einops import rearrange, repeat
from ldm.util import default

# import src_plugins.sd1111_plugin.SD_txt2img
import src_plugins.sd1111_plugin.SDPlugin
import src_plugins.sd1111_plugin.SDState
from src_plugins.sd1111_plugin import devices
from torch import einsum

from statistics import stdev, mean

hypernetworks: dict[str, Path] | None = None
hnmodel = None


class HypernetworkModule(torch.nn.Module):
    multiplier = 1.0
    activation_dict = {
        "relu"     : torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu"      : torch.nn.ELU,
        "swish"    : torch.nn.Hardswish,
    }

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, add_layer_norm=False, use_dropout=False):
        super().__init__()

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):
            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i + 1])))

            # Add an activation func
            if activation_func == "linear" or activation_func is None:
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i + 1])))

            # Add dropout expect last layer
            if use_dropout and i < len(layer_structure) - 3:
                linears.append(torch.nn.Dropout(p=0.3))

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    layer.weight.data.normal_(mean=0.0, std=0.01)
                    layer.bias.data.zero_()

        self.to(devices.device)

    def fix_old_state_dict(self, state_dict):
        changes = {
            'linear1.bias'  : 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias'  : 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x):
        return x + self.linear(x) * self.multiplier

    def trainables(self):
        layer_structure = []
        for layer in self.linear:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                layer_structure += [layer.weight, layer.bias]
        return layer_structure


def apply_strength(value=None):
    HypernetworkModule.multiplier = value if value is not None else src_plugins.sd1111_plugin.options.opts.sd_hypernetwork_strength


class Hypernetwork:
    filename = None
    name = None

    def __init__(self, name=None, enable_sizes=None, layer_structure=None, activation_func=None, add_layer_norm=False, use_dropout=False):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.layer_structure = layer_structure
        self.activation_func = activation_func
        self.add_layer_norm = add_layer_norm
        self.use_dropout = use_dropout

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.add_layer_norm, self.use_dropout),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.add_layer_norm, self.use_dropout),
            )

    def weights(self):
        res = []

        for k, layers in self.layers.items():
            for layer in layers:
                layer.train()
                res += layer.trainables()

        return res

    def save(self, filename):
        state_dict = {}

        for k, v in self.layers.items():
            state_dict[k] = (v[0].state_dict(), v[1].state_dict())

        state_dict['step'] = self.step
        state_dict['name'] = self.name
        state_dict['layer_structure'] = self.layer_structure
        state_dict['activation_func'] = self.activation_func
        state_dict['is_layer_norm'] = self.add_layer_norm
        state_dict['use_dropout'] = self.use_dropout
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name

        torch.save(state_dict, filename)

    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cpu')

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        self.activation_func = state_dict.get('activation_func', None)
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        self.use_dropout = state_dict.get('use_dropout', False)

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    HypernetworkModule(size, sd[0], self.layer_structure, self.activation_func, self.add_layer_norm, self.use_dropout),
                    HypernetworkModule(size, sd[1], self.layer_structure, self.activation_func, self.add_layer_norm, self.use_dropout),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)


def discover_hypernetworks(dirpath) -> dict[str, Path]:
    global hypernetworks
    hypernetworks = {}
    for file in dirpath.rglob('**/*.pt'):
        file = Path(file)
        name = file.stem
        hypernetworks[name] = file


def load_hypernetwork(filename):
    path = hypernetworks.get(filename, None)
    if path is not None:
        print(f"Loading hypernetwork {filename}")
        try:
            src_plugins.sd1111_plugin.SDState.hnmodel = Hypernetwork()
            src_plugins.sd1111_plugin.SDState.hnmodel.load(path)

        except Exception:
            print(f"Error loading hypernetwork {path}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
    else:
        if src_plugins.sd1111_plugin.SDState.hnmodel is not None:
            print(f"Unloading hypernetwork")

        src_plugins.sd1111_plugin.SDState.hnmodel = None


def find_hypernetwork(search: str, exact=False):
    if not search:
        return None
    if exact:
        return hypernetworks.get(search, None)

    search = search.lower()
    applicable = [name for name in hypernetworks if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return applicable[0]


def apply_hypernetwork(hypernetwork, context, layer=None):
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context.shape[2], None)

    if hypernetwork_layers is None:
        return context, context

    if layer is not None:
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    context_k = hypernetwork_layers[0](context)
    context_v = hypernetwork_layers[1](context)
    return context_k, context_v


def attention_CrossAttention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = apply_hypernetwork(src_plugins.sd1111_plugin.SDState.hnmodel, context, self)
    k = self.to_k(context_k)
    v = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


def stack_conds(conds):
    if len(conds) == 1:
        return torch.stack(conds)

    # same as in reconstruct_multicond_batch
    token_count = max([x.shape[0] for x in conds])
    for i in range(len(conds)):
        if conds[i].shape[0] != token_count:
            last_vector = conds[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - conds[i].shape[0], 1])
            conds[i] = torch.vstack([conds[i], last_vector_repeated])

    return torch.stack(conds)


def log_statistics(loss_info: dict, key, value):
    if key not in loss_info:
        loss_info[key] = [value]
    else:
        loss_info[key].append(value)
        if len(loss_info) > 1024:
            loss_info.pop(0)


def statistics(data):
    total_information = f"loss:{mean(data):.3f}" + u"\u00B1" + f"({stdev(data) / (len(data) ** 0.5):.3f})"
    recent_data = data[-32:]
    recent_information = f"recent 32 loss:{mean(recent_data):.3f}" + u"\u00B1" + f"({stdev(recent_data) / (len(recent_data) ** 0.5):.3f})"
    return total_information, recent_information


def report_statistics(loss_info: dict):
    keys = sorted(loss_info.keys(), key=lambda x: sum(loss_info[x]) / len(loss_info[x]))
    for key in keys:
        try:
            print("Loss statistics for file " + key)
            info, recent = statistics(loss_info[key])
            print(info)
            print(recent)
        except Exception as e:
            print(e)


