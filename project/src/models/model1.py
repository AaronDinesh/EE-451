import torch
import torch.nn as nn
from src.modules.ConvLayer import ConvLayer
from src.modules.BottleneckLayer import BottleneckLayer
from src.modules.C2F import C2F
from src.modules.SPPF import SPPF
from src.modules.Detect import Detect


class Concat(nn.Module):
    """Concatenate a list of tensors along a given dimension"""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)

# Hardcoded YOLOv8 configuration
CONFIG = {
    'nc': 80,  # number of classes
    'scales': {
        'n': (0.33, 0.25, 1024),
        's': (0.33, 0.50, 1024),
        'm': (0.67, 0.75, 768),
        'l': (1.00, 1.00, 512),
        'x': (1.00, 1.25, 512),
    },
    'backbone': [
        [-1, 1, 'Conv', [64, 3, 2]],
        [-1, 1, 'Conv', [128, 3, 2]],
        [-1, 3, 'C2f', [128, True]],
        [-1, 1, 'Conv', [256, 3, 2]],
        [-1, 6, 'C2f', [256, True]],
        [-1, 1, 'Conv', [512, 3, 2]],
        [-1, 6, 'C2f', [512, True]],
        [-1, 1, 'Conv', [1024, 3, 2]],
        [-1, 3, 'C2f', [1024, True]],
        [-1, 1, 'SPPF', [1024, 5]],
    ],
    'head': [
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 6], 1, 'Concat', [1]],
        [-1, 3, 'C2f', [512]],
        [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
        [[-1, 4], 1, 'Concat', [1]],
        [-1, 3, 'C2f', [256]],
        [-1, 1, 'Conv', [256, 3, 2]],
        [[-1, 12], 1, 'Concat', [1]],
        [-1, 3, 'C2f', [512]],
        [-1, 1, 'Conv', [512, 3, 2]],
        [[-1, 9], 1, 'Concat', [1]],
        [-1, 3, 'C2f', [1024]],
        [[15, 18, 21], 1, 'Detect', ['nc']],
    ],
}

class Model(nn.Module):
    """
    YOLOv8 model with hardcoded configuration.
    Default variant is 'n' (nano). Other variants: 's', 'm', 'l', 'x'.
    """
    def __init__(self, variant='n', ch=3, nc=None):
        super().__init__()
        # Model hyperparameters
        d_mul, w_mul, _ = CONFIG['scales'][variant]
        self.nc = nc or CONFIG['nc']
        # Combine backbone and head layers
        layers_cfg = CONFIG['backbone'] + CONFIG['head']
        self.layers = nn.ModuleList()
        self.layers_cfg = []
        chs = [ch]

        for idx, (f, n, m, args) in enumerate(layers_cfg):
            # Resolve module class
            if isinstance(m, str):
                if m == 'Conv':
                    module_cls = ConvLayer
                elif m == 'C2f' or m == 'C2F':
                    module_cls = C2F
                elif m == 'SPPF':
                    module_cls = SPPF
                elif m == 'Detect':
                    module_cls = Detect
                elif m == 'Concat':
                    module_cls = Concat
                elif m == 'nn.Upsample':
                    module_cls = nn.Upsample
                else:
                    raise ValueError(f"Unknown module type '{m}'")
            else:
                module_cls = m

            # Compute number of repeats
            n_repeats = max(round(n * d_mul), 1) if n > 1 else n
            module_kwargs = {}

            if module_cls is ConvLayer:
                out_ch, k, s = args
                out_ch = max(round(out_ch * w_mul), 1)
                padding = k // 2
                module_args = (chs[f], out_ch, k, s, padding)
                out_channels = out_ch

            elif module_cls is C2F:
                out_ch, shortcut = args
                out_ch = max(round(out_ch * w_mul), 1)
                module_args = (shortcut, chs[f], out_ch, n)
                out_channels = out_ch
                n_repeats = 1

            elif module_cls is SPPF:
                out_ch, k = args
                out_ch = max(round(out_ch * w_mul), 1)
                module_args = (chs[f], out_ch, k)
                out_channels = out_ch

            elif module_cls is Concat:
                module_args = (args[0],)
                out_channels = chs[f] if not isinstance(args[0], list) else sum(chs[j] for j in args[0])

            elif module_cls is nn.Upsample:
                _, scale_factor, mode = args
                module_args = ()
                module_kwargs = {'size': None, 'scale_factor': scale_factor, 'mode': mode}
                out_channels = chs[f]

            elif module_cls is Detect:
                module_args = (self.nc, [chs[j] for j in f])
                out_channels = None
                n_repeats = 1

            else:
                module_args = args if isinstance(args, list) else []
                out_channels = chs[f]

            # Build module(s)
            modules = [module_cls(*module_args, **module_kwargs) for _ in range(n_repeats)]
            layer = nn.Sequential(*modules) if len(modules) > 1 else modules[0]

            self.layers.append(layer)
            self.layers_cfg.append(f)
            chs.append(out_channels)

        self.chs = chs

    def forward(self, x):
        outputs = [x]
        y = None
        for f, layer in zip(self.layers_cfg, self.layers):
            inp = [outputs[i] for i in f] if isinstance(f, list) else outputs[f]
            y = layer(inp) if isinstance(inp, list) else layer(inp)
            outputs.append(y)
        return y


if __name__ == '__main__':
    model = Model(variant='n')
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print("Output shape:", y.shape if isinstance(y, torch.Tensor) else [t.shape for t in y])