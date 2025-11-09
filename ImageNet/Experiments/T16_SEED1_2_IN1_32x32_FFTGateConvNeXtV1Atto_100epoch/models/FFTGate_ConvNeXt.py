


# %%  


'''
ConvNeXt in PyTorch.
'''




# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# ✅ Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

import sys
import os



# ✅ Import custom activation function
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from activation.FFTGate import FFTGate # type: ignore









# ===========================
# FFTGate adapter NHWC <-> NCHW
# ===========================

# ===== FFTGateAdapter (no NHWC↔NCHW permutes; stays channels-last end-to-end) =====
import torch.nn as nn

class FFTGateAdapter(nn.Module):
    def __init__(self, num_channels, **kwargs):
        super().__init__()
        self.core = FFTGate(num_channels=num_channels, **kwargs)

    def forward(self, x, epoch=0):
        if epoch is None:
            epoch = 0
        return self.core(x, epoch=epoch)  # 🟡 passthrough; no permutes











class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers



        # self.act = nn.GELU()
        self.act = FFTGateAdapter(                  # 🟡 provide C=4*dim
            num_channels=4 * dim,                   # 🟡 per-channel params live here
            gamma1_init=1.5,
            freq_init=0.3,
            phi=0.1,
            history_len=16,                         # ✅ keeps last 16 batches
            decay_mode="linear"
        )




        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, epoch=None):  # # 🟡 accept epoch and propagate to FFTGate
        if epoch is None:
            epoch = 0

    # def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)

        x = self.act(x, epoch=epoch)  # # 🟡 pass epoch to FFTGate)
        # x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000  --> change from 1000 (ImageNet-1K) to 100 (ImageNet-100)
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        # ✅ Stem layer        
        # stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        # self.downsample_layers.append(stem)

        # 🟡 MODIFIED for 32×32 inputs (was kernel_size=4, stride=4)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),  # 🟡 keep resolution 32×32
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)


        # ✅ Downsampling layers between stages        
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),    # 🟡 standard stride-2 downsamples
            )
            self.downsample_layers.append(downsample_layer)

        # ✅ Stages (residual blocks per stage)
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # ✅ Final normalization and classification head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)







    # ✅ Returns full feature maps (before GAP + Norm)
    def forward_features_map(self, x, epoch=None):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for blk in self.stages[i]:
                x = blk(x, epoch=epoch)   # epoch flows into FFTGate
        return x  # (N, C, H, W)


    # ✅ Returns pooled + normalized features (default ConvNeXt behavior)
    def forward_features(self, x, epoch=None):
        x = self.forward_features_map(x, epoch=epoch)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    # ✅ Classification output
    # def forward(self, x, epoch=None):
    #     x = self.forward_features(x, epoch=epoch)
    #     x = self.head(x)
    #     return x     # shape: (N, num_classes)
    def forward(self, x, epoch=None):
        feats = self.forward_features(x, epoch=epoch)  # (N, C) after norm
        out = self.head(feats)                         # logits
        return out, feats                              # always return both













class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model







@register_model
def convnextv1_atto(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    if pretrained:
        raise NotImplementedError("No official pretrained weights for ConvNeXt V1-A (Atto).")
    return model












# %%  



 


# # ================== 0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ ================================================================ 
# ########################################################################################################################
# ####-------| NOTE 0️⃣ DEFINE MODEL AND ACTIVATION FUCNTION | XXX -----------------------------------####################
# ########################################################################################################################

# # ✅ --- Load Libraries ---
# import torch
# import platform, sys
# from ptflops import get_model_complexity_info  # pip install ptflops
# from contextlib import nullcontext


# # ✅ --- Config ---
# MODEL_NAME = "ConvNeXt-V1-A"
# ACT_NAME   = "FFTGate"
# NUM_CLASSES = 1000   # 100 for IN-100, 1000 for IN-1K
# INPUT_RES = 32
# INPUT_SIZE = (3, INPUT_RES, INPUT_RES)

# INPUT_RES_str = str(INPUT_RES) # Convert to string  for filenames


# # ✅ --- pretty header like your screenshot ---
# header_text = f" {MODEL_NAME} | {ACT_NAME} | Resolution → {INPUT_RES_str}x{INPUT_RES_str} "
# header_line = f"{header_text:=^70}"   # centered, filled with '='; tweak width if you want




# ########################################################################################################################
# ####-------| NOTE 1️⃣ SANITY CHECK | XXX ------------------------------------------------------------####################
# ########################################################################################################################
# # --- Build model on GPU for sanity / speed ---
# assert torch.cuda.is_available(), "CUDA is required for this run."
# device = "cuda"
# net = convnextv1_atto(pretrained=False, num_classes=NUM_CLASSES).eval().to(device)

# x = torch.randn(2, *INPUT_SIZE, device=device)
# with torch.no_grad():
#     logits, feats = net(x, epoch=0)   # unpack both
# print("logits:", tuple(logits.shape), "| feats:", tuple(feats.shape),
#       "| device:", next(net.parameters()).device)
# # expect logits → (2, NUM_CLASSES), feats → (2, C)






# ########################################################################################################################
# ####-------| NOTE 2️⃣ COMPLEXITY CHECK | XXX --------------------------------------------------------####################
# ########################################################################################################################

# # ✅ Measure Params (M) and FLOPs (G) for ConvNeXt-T | Params (M) & FLOPs (G, reported as GMACs to match paper)



# # wrapper so ptflops can call model(x) without an epoch arg
# class InferenceWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model.eval()
#     def forward(self, x):
#         with torch.no_grad():
#             logits, _ = self.model(x, epoch=0)   # return only logits
#             return logits



# device = "cuda"
# model = convnextv1_atto(pretrained=False, num_classes=NUM_CLASSES).to(device)
# wrapped = InferenceWrapper(model).to(device)


# with torch.no_grad():
#     macs, params = get_model_complexity_info(
#         wrapped,
#         input_res=INPUT_SIZE,
#         as_strings=False,
#         print_per_layer_stat=False,
#         verbose=False,
#     )

# params_m = params / 1e6
# gmacs = macs / 1e9  # already MACs, so this is the number papers call "FLOPs (G)"
# print(f"Params (M): {params_m:.2f}")
# print(f'FLOPs (G): {gmacs:.2f}')  # expect ~29.0M, ~4.5G for ConvNeXt-T @224²










# ########################################################################################################################
# ####-------| NOTE 3️⃣ Inference Speed & MEMORY | XXX ------------------------------------------------####################
# ########################################################################################################################
# # Inference speed & memory (GPU-only)




# def measure_inference(model, input_size=INPUT_SIZE, epoch=0,
#                       precision="fp16", iters=100, warmup=20):
#     assert torch.cuda.is_available(), "CUDA required"
#     device = "cuda"
#     model = model.to(device).eval()
#     torch.backends.cudnn.benchmark = True

#     amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
#                if precision.lower() == "fp16" else nullcontext())

#     # ---------- Throughput @ bs=32 ----------
#     bs32 = 32
#     x32 = torch.randn(bs32, *input_size, device=device)

#     # warmup
#     for _ in range(warmup):
#         with torch.no_grad():
#             with amp_ctx:
#                 logits, _ = model(x32, epoch=epoch)   # ✅ unpack, ignore feats

#     torch.cuda.synchronize()
#     start = torch.cuda.Event(True); end = torch.cuda.Event(True)
#     start.record()
#     for _ in range(iters):
#         with torch.no_grad():
#             with amp_ctx:
#                 logits, _ = model(x32, epoch=epoch)   # ✅ unpack
#     end.record(); torch.cuda.synchronize()
#     total_ms = start.elapsed_time(end)
#     throughput = (iters * bs32) / (total_ms / 1000.0)  # img/s

#     # Peak mem @ bs=32 (inference)
#     torch.cuda.reset_peak_memory_stats()
#     with torch.no_grad():
#         with amp_ctx:
#             logits, _ = model(x32, epoch=epoch)       # ✅ unpack
#     torch.cuda.synchronize()
#     peak_mem_bs32_gb = torch.cuda.max_memory_allocated() / 1e9

#     # ---------- Latency @ bs=1 ----------
#     x1 = torch.randn(1, *input_size, device=device)
#     for _ in range(warmup):
#         with torch.no_grad():
#             with amp_ctx:
#                 logits, _ = model(x1, epoch=epoch)    # ✅ unpack

#     torch.cuda.synchronize()
#     start = torch.cuda.Event(True); end = torch.cuda.Event(True)
#     start.record()
#     for _ in range(iters):
#         with torch.no_grad():
#             with amp_ctx:
#                 logits, _ = model(x1, epoch=epoch)    # ✅ unpack
#     end.record(); torch.cuda.synchronize()
#     latency_ms = start.elapsed_time(end) / iters  # ms/img

#     # Peak mem @ bs=1 (inference)
#     torch.cuda.reset_peak_memory_stats()
#     with torch.no_grad():
#         with amp_ctx:
#             logits, _ = model(x1, epoch=epoch)        # ✅ unpack
#     torch.cuda.synchronize()
#     peak_mem_bs1_gb = torch.cuda.max_memory_allocated() / 1e9

#     return throughput, latency_ms, peak_mem_bs32_gb, peak_mem_bs1_gb


# # --- Usage ---
# net = convnextv1_atto(pretrained=False, num_classes=NUM_CLASSES)
# thr, lat, mem_bs32, mem_bs1 = measure_inference(net, INPUT_SIZE, epoch=0, precision="fp16")  # or "fp32"
# print(f"Throughput (img/s, bs=32): {thr:.1f}")
# print(f"Latency (ms/img, bs=1): {lat:.2f}")
# print(f"Peak Mem (GB, inf., bs=32): {mem_bs32:.2f}")
# print(f"Peak Mem (GB, inf., bs=1):  {mem_bs1:.2f}")



 




# ########################################################################################################################
# ####-------| NOTE 4️⃣ SUMMARY REPORT | XXX ----------------------------------------------------------####################
# ########################################################################################################################

# # ✅ --- Define path to store SUMMARY REPORT ---
# save_paths = {
#     "summaryreport_log_history": f"C:\\Users\\emeka\\Research\\ModelCUDA\\Big_Data_Journal\\Comparison\\Code\\Paper\\github3\\ImageNet_1k\ConvNeXt_V1_Atto\\Results\\FFTGate\\Logs\\Summary_Report_{MODEL_NAME}_{ACT_NAME}_logs.txt"
# }

# top1 = None  # fill with your val accuracy when ready

# # ✅ --- summary_line ---
# summary_line = (f"{MODEL_NAME} | {ACT_NAME} | Top-1: "
#                 f"{'—' if top1 is None else f'{top1:.2f}'} | "
#                 f"Params (M): {params_m:.2f} | FLOPs (G): {gmacs:.2f} | "
#                 f"Thru (bs=32): {thr:.1f} | Lat (bs=1): {lat:.2f} | "
#                 f"Peak Mem (GB, inf., bs=32): {mem_bs32:.2f} | "
#                 f"Peak Mem (GB, inf., bs=1): {mem_bs1:.2f}")


# # ✅ --- environment_line ---
# props = torch.cuda.get_device_properties(0)
# env_summary = (
#     f"GPU: {torch.cuda.get_device_name(0)} ({props.total_memory/1e9:.2f} GB)\n"
#     f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}\n"
#     f"Python: {sys.version.split()[0]} | OS: {platform.system()} {platform.release()}\n"
# )


# # ✅ --- print to console ---
# print(header_line)
# print(env_summary)
# print(summary_line)



# # ✅ --- Create folder if it doesn't exist, then write ---
# log_path = save_paths["summaryreport_log_history"]
# os.makedirs(os.path.dirname(log_path), exist_ok=True)
# with open(log_path, "w", encoding="utf-8") as f:   # open the file with "w" to overwrite it on each run. "a" appends; "w" truncates and writes fresh
#     f.write(header_line + "\n")
#     f.write(env_summary + "\n")
#     f.write(summary_line + "\n")



# # %%
