from collections import OrderedDict
from copy import deepcopy
import json
import warnings
import os
import random

import torch
import numpy as np

def slprint(x, name='x'):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        print(f'{name}.shape:', x.shape)
    elif isinstance(x, (tuple, list)):
        print('type x:', type(x))
        for i in range(min(10, len(x))):
            slprint(x[i], f'{name}[{i}]')
    elif isinstance(x, dict):
        for k,v in x.items():
            slprint(v, f'{name}[{k}]')
    else:
        print(f'{name}.type:', type(x))

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict

def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)



class CocoClassMapper():
    def __init__(self) -> None:
        self.category_map_str = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "13": 12, "14": 13, "15": 14, "16": 15, "17": 16, "18": 17, "19": 18, "20": 19, "21": 20, "22": 21, "23": 22, "24": 23, "25": 24, "27": 25, "28": 26, "31": 27, "32": 28, "33": 29, "34": 30, "35": 31, "36": 32, "37": 33, "38": 34, "39": 35, "40": 36, "41": 37, "42": 38, "43": 39, "44": 40, "46": 41, "47": 42, "48": 43, "49": 44, "50": 45, "51": 46, "52": 47, "53": 48, "54": 49, "55": 50, "56": 51, "57": 52, "58": 53, "59": 54, "60": 55, "61": 56, "62": 57, "63": 58, "64": 59, "65": 60, "67": 61, "70": 62, "72": 63, "73": 64, "74": 65, "75": 66, "76": 67, "77": 68, "78": 69, "79": 70, "80": 71, "81": 72, "82": 73, "84": 74, "85": 75, "86": 76, "87": 77, "88": 78, "89": 79, "90": 80}
        self.origin2compact_mapper = {int(k):v-1 for k,v in self.category_map_str.items()}
        self.compact2origin_mapper = {int(v-1):int(k) for k,v in self.category_map_str.items()}

    def origin2compact(self, idx):
        return self.origin2compact_mapper[int(idx)]

    def compact2origin(self, idx):
        return self.compact2origin_mapper[int(idx)]

def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, list):
        return [to_device(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to_device(v, device) for k,v in item.items()}
    else:
        raise NotImplementedError("Call Shilong if you use other containers! type: {}".format(type(item)))



# 
def get_gaussian_mean(x, axis, other_axis, softmax=True):
    """

    Args:
        x (float): Input images(BxCxHxW)
        axis (int): The index for weighted mean
        other_axis (int): The other index

    Returns: weighted index for axis, BxC

    """
    mat2line = torch.sum(x, axis=other_axis)
    # mat2line = mat2line / mat2line.mean() * 10
    if softmax:
        u = torch.softmax(mat2line, axis=2)
    else:
        u = mat2line / (mat2line.sum(2, keepdim=True) + 1e-6)
    size = x.shape[axis]
    ind = torch.linspace(0, 1, size).to(x.device)
    batch = x.shape[0]
    channel = x.shape[1]
    index = ind.repeat([batch, channel, 1])
    mean_position = torch.sum(index * u, dim=2)
    return mean_position

def get_expected_points_from_map(hm, softmax=True):
    """get_gaussian_map_from_points
        B,C,H,W -> B,N,2 float(0, 1) float(0, 1)
        softargmax function

    Args:
        hm (float): Input images(BxCxHxW)

    Returns: 
        weighted index for axis, BxCx2. float between 0 and 1.

    """
    # hm = 10*hm
    B,C,H,W = hm.shape
    y_mean = get_gaussian_mean(hm, 2, 3, softmax=softmax) # B,C
    x_mean = get_gaussian_mean(hm, 3, 2, softmax=softmax) # B,C
    # return torch.cat((x_mean.unsqueeze(-1), y_mean.unsqueeze(-1)), 2)
    return torch.stack([x_mean, y_mean], dim=2)

# Positional encoding (section 5.1)
# borrow from nerf
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    import torch.nn as nn
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class APOPMeter():
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update(self, pred, gt):
        """
        Input:
            pred, gt: Tensor()
        """
        assert pred.shape == gt.shape
        self.tp += torch.logical_and(pred == 1, gt == 1).sum().item()
        self.fp += torch.logical_and(pred == 1, gt == 0).sum().item()
        self.tn += torch.logical_and(pred == 0, gt == 0).sum().item()
        self.tn += torch.logical_and(pred == 1, gt == 0).sum().item()

    def update_cm(self, tp, fp, tn, fn):
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.tn += fn

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

import argparse
from util.slconfig import SLConfig
def get_raw_dict(args):
    """
    return the dicf contained in args.
    
    e.g:
        >>> with open(path, 'w') as f:
                json.dump(get_raw_dict(args), f, indent=2)
    """
    if isinstance(args, argparse.Namespace): 
        return vars(args)   
    elif isinstance(args, dict):
        return args
    elif isinstance(args, SLConfig):
        return args._cfg_dict
    else:
        raise NotImplementedError("Unknown type {}".format(type(args)))


def stat_tensors(tensor):
    assert tensor.dim() == 1
    tensor_sm = tensor.softmax(0)
    entropy = (tensor_sm * torch.log(tensor_sm + 1e-9)).sum()

    return {
        'max': tensor.max(),
        'min': tensor.min(),
        'mean': tensor.mean(),
        'var': tensor.var(),
        'std': tensor.var() ** 0.5,
        'entropy': entropy
    }


class NiceRepr:
    """Inherit from this class and define ``__nice__`` to "nicely" print your
    objects.

    Defines ``__str__`` and ``__repr__`` in terms of ``__nice__`` function
    Classes that inherit from :class:`NiceRepr` should redefine ``__nice__``.
    If the inheriting class has a ``__len__``, method then the default
    ``__nice__`` method will return its length.

    Example:
        >>> class Foo(NiceRepr):
        ...    def __nice__(self):
        ...        return 'info'
        >>> foo = Foo()
        >>> assert str(foo) == '<Foo(info)>'
        >>> assert repr(foo).startswith('<Foo(info) at ')

    Example:
        >>> class Bar(NiceRepr):
        ...    pass
        >>> bar = Bar()
        >>> import pytest
        >>> with pytest.warns(None) as record:
        >>>     assert 'object at' in str(bar)
        >>>     assert 'object at' in repr(bar)

    Example:
        >>> class Baz(NiceRepr):
        ...    def __len__(self):
        ...        return 5
        >>> baz = Baz()
        >>> assert str(baz) == '<Baz(5)>'
    """

    def __nice__(self):
        """str: a "nice" summary string describing this module"""
        if hasattr(self, '__len__'):
            # It is a common pattern for objects to use __len__ in __nice__
            # As a convenience we define a default __nice__ for these objects
            return str(len(self))
        else:
            # In all other cases force the subclass to overload __nice__
            raise NotImplementedError(
                f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        """str: the string of the module"""
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        """str: the string of the module"""
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)



def ensure_rng(rng=None):
    """Coerces input into a random number generator.

    If the input is None, then a global random state is returned.

    If the input is a numeric value, then that is used as a seed to construct a
    random state. Otherwise the input is returned as-is.

    Adapted from [1]_.

    Args:
        rng (int | numpy.random.RandomState | None):
            if None, then defaults to the global rng. Otherwise this can be an
            integer or a RandomState class
    Returns:
        (numpy.random.RandomState) : rng -
            a numpy random number generator

    References:
        .. [1] https://gitlab.kitware.com/computer-vision/kwarray/blob/master/kwarray/util_random.py#L270  # noqa: E501
    """

    if rng is None:
        rng = np.random.mtrand._rand
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)
    else:
        rng = rng
    return rng

def random_boxes(num=1, scale=1, rng=None):
    """Simple version of ``kwimage.Boxes.random``

    Returns:
        Tensor: shape (n, 4) in x1, y1, x2, y2 format.

    References:
        https://gitlab.kitware.com/computer-vision/kwimage/blob/master/kwimage/structs/boxes.py#L1390

    Example:
        >>> num = 3
        >>> scale = 512
        >>> rng = 0
        >>> boxes = random_boxes(num, scale, rng)
        >>> print(boxes)
        tensor([[280.9925, 278.9802, 308.6148, 366.1769],
                [216.9113, 330.6978, 224.0446, 456.5878],
                [405.3632, 196.3221, 493.3953, 270.7942]])
    """
    rng = ensure_rng(rng)

    tlbr = rng.rand(num, 4).astype(np.float32)

    tl_x = np.minimum(tlbr[:, 0], tlbr[:, 2])
    tl_y = np.minimum(tlbr[:, 1], tlbr[:, 3])
    br_x = np.maximum(tlbr[:, 0], tlbr[:, 2])
    br_y = np.maximum(tlbr[:, 1], tlbr[:, 3])

    tlbr[:, 0] = tl_x * scale
    tlbr[:, 1] = tl_y * scale
    tlbr[:, 2] = br_x * scale
    tlbr[:, 3] = br_y * scale

    boxes = torch.from_numpy(tlbr)
    return boxes


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class BestMetricSingle():
    def __init__(self, init_res=0.0, better='large') -> None:
        self.init_res = init_res
        self.best_res = init_res
        self.best_ep = -1

        self.better = better
        assert better in ['large', 'small']

    def isbetter(self, new_res, old_res):
        if self.better == 'large':
            return new_res > old_res
        if self.better == 'small':
            return new_res < old_res

    def update(self, new_res, ep):
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def __str__(self) -> str:
        return "best_res: {}\t best_ep: {}".format(self.best_res, self.best_ep)

    def __repr__(self) -> str:
        return self.__str__()

    def summary(self) -> dict:
        return {
            'best_res': self.best_res,
            'best_ep': self.best_ep,
        }


class BestMetricHolder():
    def __init__(self, init_res=0.0, better='large', use_ema=False) -> None:
        self.best_all = BestMetricSingle(init_res, better)
        self.use_ema = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_res, better)
            self.best_regular = BestMetricSingle(init_res, better)
    

    def update(self, new_res, epoch, is_ema=False):
        """
        return if the results is the best.
        """
        if not self.use_ema:
            return self.best_all.update(new_res, epoch)
        else:
            if is_ema:
                self.best_ema.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)
            else:
                self.best_regular.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)

    def summary(self):
        if not self.use_ema:
            return self.best_all.summary()

        res = {}
        res.update({f'all_{k}':v for k,v in self.best_all.summary().items()})
        res.update({f'regular_{k}':v for k,v in self.best_regular.summary().items()})
        res.update({f'ema_{k}':v for k,v in self.best_ema.summary().items()})
        return res

    def __repr__(self) -> str:
        return json.dumps(self.summary(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------
def seed_all(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed python, numpy, torch (cpu & cuda) and configure deterministic flags.
    Safe to call multiple times.
    """
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = False
            cudnn.deterministic = True
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker_init_fn to make dataloader transforms deterministic.
    Uses torch.initial_seed to derive a distinct seed per worker.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def valid_ratios_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute valid ratios (w_ratio, h_ratio) from a binary padding mask.
    """
    if mask.dim() != 3:
        raise ValueError(f"mask must be [B,H,W], got {tuple(mask.shape)}")
    _, h, w = mask.shape
    valid_h = torch.sum(~mask[:, :, 0], 1)
    valid_w = torch.sum(~mask[:, 0, :], 1)
    vr_h = valid_h.float() / float(h)
    vr_w = valid_w.float() / float(w)
    return torch.stack([vr_w, vr_h], dim=-1)


def build_box_aware_density_targets(
    targets: list,
    mask_lvl0: torch.Tensor,
    sigma_factor: float = 0.25,
    sigma_min: float = 1.0,
    radius_factor: float = 3.0,
    rmax: int = 16,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Build box-aware density targets at a given (level-0) resolution.
    """
    if mask_lvl0.dim() != 3:
        raise ValueError(
            f"mask_lvl0 must be [B,H0,W0], got {tuple(mask_lvl0.shape)}"
        )
    device = mask_lvl0.device
    B, H0, W0 = mask_lvl0.shape
    density = torch.zeros((B, 1, H0, W0), device=device, dtype=torch.float32)
    valid_ratios = valid_ratios_from_mask(mask_lvl0).to(
        device=device, dtype=torch.float32
    )

    for b in range(B):
        if not isinstance(targets[b], dict) or 'boxes' not in targets[b]:
            continue
        boxes = targets[b]['boxes']
        if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
            continue
        boxes = boxes.to(device=device, dtype=torch.float32)

        valid_h = int((~mask_lvl0[b, :, 0]).sum().item())
        valid_w = int((~mask_lvl0[b, 0, :]).sum().item())
        valid_h = max(1, min(valid_h, H0))
        valid_w = max(1, min(valid_w, W0))

        vr_w, vr_h = valid_ratios[b, 0], valid_ratios[b, 1]
        scale = torch.tensor(
            [vr_w, vr_h, vr_w, vr_h], device=device, dtype=torch.float32
        )
        boxes_pad = boxes * scale[None, :]

        cx = boxes_pad[:, 0].clamp(0.0, 1.0) * float(W0) - 0.5
        cy = boxes_pad[:, 1].clamp(0.0, 1.0) * float(H0) - 0.5
        bw = boxes_pad[:, 2].clamp_min(0.0) * float(W0)
        bh = boxes_pad[:, 3].clamp_min(0.0) * float(H0)

        sigma_x = (bw * float(sigma_factor)).clamp_min(float(sigma_min))
        sigma_y = (bh * float(sigma_factor)).clamp_min(float(sigma_min))

        rx = torch.ceil(float(radius_factor) * sigma_x).clamp(
            max=float(rmax)
        ).to(torch.long)
        ry = torch.ceil(float(radius_factor) * sigma_y).clamp(
            max=float(rmax)
        ).to(torch.long)
        rx_max = int(rx.max().item())
        ry_max = int(ry.max().item())
        rx_max = max(0, rx_max)
        ry_max = max(0, ry_max)

        if rx_max == 0 and ry_max == 0:
            xi = torch.round(cx).to(torch.long).clamp(0, valid_w - 1)
            yi = torch.round(cy).to(torch.long).clamp(0, valid_h - 1)
            flat_idx = (yi * W0 + xi).clamp(0, H0 * W0 - 1)
            density[b, 0].view(-1).scatter_add_(
                0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
            )
            continue

        dx = torch.arange(-rx_max, rx_max + 1, device=device, dtype=torch.long)
        dy = torch.arange(-ry_max, ry_max + 1, device=device, dtype=torch.long)
        gy, gx = torch.meshgrid(dy, dx, indexing='ij')
        dx_flat = gx.reshape(-1)
        dy_flat = gy.reshape(-1)

        cx_int = torch.round(cx).to(torch.long)
        cy_int = torch.round(cy).to(torch.long)
        x_idx = cx_int[:, None] + dx_flat[None, :]
        y_idx = cy_int[:, None] + dy_flat[None, :]

        in_rect = (x_idx >= 0) & (x_idx < valid_w) & (y_idx >= 0) & (y_idx < valid_h)
        in_rad = (dx_flat.abs()[None, :] <= rx[:, None]) & (
            dy_flat.abs()[None, :] <= ry[:, None]
        )
        keep = in_rect & in_rad

        dx_f = x_idx.to(torch.float32) - cx[:, None]
        dy_f = y_idx.to(torch.float32) - cy[:, None]
        w = torch.exp(
            -0.5
            * (
                (dx_f / (sigma_x[:, None] + eps)) ** 2
                + (dy_f / (sigma_y[:, None] + eps)) ** 2
            )
        )
        w = w * keep.to(torch.float32)

        w_sum = w.sum(dim=1, keepdim=True).clamp_min(eps)
        w = w / w_sum

        flat = (y_idx * W0 + x_idx).clamp(0, H0 * W0 - 1)
        density[b, 0].view(-1).scatter_add_(
            0, flat.reshape(-1), w.reshape(-1)
        )

    return density
            
