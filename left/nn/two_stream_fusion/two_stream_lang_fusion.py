#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : two_stream_attention_lang_fusion.py
# Author : Joy Hsu
# Email  : joycj@stanford.edu
# Date   : 04/04/2023
#
# This file is part of Project Concepts.
# Distributed under terms of the MIT license.


import numpy as np
import torch
import torch.nn.functional as F

from cliport.models.core.attention import Attention # Potentially move to concepts
from cliport.models.core.transport import Transport
import cliport.models as models
import cliport.models.core.fusion as fusion


class TwoStreamAttentionLangFusion(Attention):
    """Two Stream Language-Conditioned Attention (a.k.a Pick) module."""
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.attn_stream_one = stream_one_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.attn_stream_two = stream_two_model(self.in_shape, 1, self.cfg, self.device, self.preprocess)
        self.fusion = fusion.names[self.fusion_type](input_dim=1)

        print(f"Attn FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def attend(self, x, l):
        x1 = self.attn_stream_one(x)
        x2 = self.attn_stream_two(x, l)
        x = self.fusion(x1, x2)
        return x

    def forward(self, inp_img, lang_goal, softmax=True):
        """Forward pass."""
        # inp_img = (320, 160, 6)
        if isinstance(inp_img, torch.Tensor):
            in_data = F.pad(inp_img, self.padding.reshape(-1).tolist(), mode='constant')    # (320, 320, 6)
            in_shape = (1,) + in_data.shape                             # (1, 320, 320, 6)
            in_data = in_data.reshape(in_shape)                         # (1, 320, 320, 6)
        else:
            in_data = np.pad(inp_img, self.padding, mode='constant')    # (320, 320, 6)
            in_shape = (1,) + in_data.shape                             # (1, 320, 320, 6)
            in_data = in_data.reshape(in_shape)                         # (1, 320, 320, 6)
            in_data = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]

        # Rotation pivot (i.e. rotation center)
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_data = in_data.permute(0, 3, 1, 2)  # [B 6 W H]
        in_data = in_data.repeat(self.n_rotations, 1, 1, 1)
        in_data = self.rotator(in_data, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_data:
            lgts = self.attend(x, lang_goal)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)                           # (1, 1, 320, 320)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]             # Retrieve original (unpadded) logits

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))           # Softmax over the entire featuremap
        if softmax:
            output = F.softmax(output, dim=-1)
            # output = output.reshape(logits.shape[1:])
        return 
    
    
class TwoStreamAttentionRepeatConcatLangFusionLatNS(TwoStreamAttentionLangFusion):
    """Language-Conditioned Attention (a.k.a Pick) module with lateral connections."""
    def __init__(self, stream_fcn, in_shape, n_rotations, preprocess, cfg, device):
        self.fusion_type = cfg['train']['attn_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, preprocess, cfg, device)


    def forward(self, inp_img, prev_map, lang_goal, softmax=True):
        """Forward pass."""
        # inp_img = (320, 160, 6)
        if isinstance(inp_img, torch.Tensor):
            in_data = F.pad(inp_img, self.padding.reshape(-1).tolist(), mode='constant')    # (320, 320, 6)
            in_shape = (1,) + in_data.shape                             # (1, 320, 320, 6)
            in_data = in_data.reshape(in_shape)                         # (1, 320, 320, 6)
        else:
            in_data = np.pad(inp_img, self.padding, mode='constant')    # (320, 320, 6)
            in_shape = (1,) + in_data.shape                             # (1, 320, 320, 6)
            in_data = in_data.reshape(in_shape)                         # (1, 320, 320, 6)
            in_data = torch.from_numpy(in_data).to(dtype=torch.float, device=self.device)  # [B W H 6]
    
        # prev_map = (320, 160, 1), torch.Tensor
        map_data = F.pad(prev_map, self.padding.reshape(-1).tolist(), mode='constant')    # (320, 320, 1)
        map_data = map_data.reshape((1,) + map_data.shape)                                # (1, 320, 320, 1)

        # Concatenate
        in_data = torch.cat((in_data, map_data), dim=-1)

        # Rotation pivot (i.e. rotation center)
        pv = np.array(in_data.shape[1:3]) // 2

        # Rotate input.
        in_data = in_data.permute(0, 3, 1, 2)  # [B 7 W H]
        in_data = in_data.repeat(self.n_rotations, 1, 1, 1)
        in_data = self.rotator(in_data, pivot=pv)

        # Forward pass.
        logits = []
        for x in in_data:
            # Retrieve last channel --> this is the (rotated) filter information
            f = x[:, -1, :, :].unsqueeze(1)
            lgts = self.attend(x, f, lang_goal)
            logits.append(lgts)
        logits = torch.cat(logits, dim=0)

        # Rotate back output.
        logits = self.rotator(logits, reverse=True, pivot=pv)
        logits = torch.cat(logits, dim=0)                           # (1, 1, 320, 320)
        c0 = self.padding[:2, 0]
        c1 = c0 + inp_img.shape[:2]
        logits = logits[:, :, c0[0]:c1[0], c0[1]:c1[1]]             # Retrieve original (unpadded) logits

        logits = logits.permute(1, 2, 3, 0)  # [B W H 1]
        output = logits.reshape(1, np.prod(logits.shape))           # Softmax over the entire featuremap
        if softmax:
            output = F.softmax(output, dim=-1)
            output = output.reshape(logits.shape[1:])

        return output

    def attend(self, x, f, l):
        # import ipdb; ipdb.set_trace()

        x1, lat = self.attn_stream_one(x, f)
        x2 = self.attn_stream_two(x, lat, f, l)
        x = self.fusion(x1, x2)
        return x
    
    
class TwoStreamTransportLangFusion(Transport):
    """Two Stream Transport (a.k.a Place) module"""
    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):
        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def _build_nets(self):
        stream_one_fcn, stream_two_fcn = self.stream_fcn
        stream_one_model = models.names[stream_one_fcn]
        stream_two_model = models.names[stream_two_fcn]

        self.key_stream_one = stream_one_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.key_stream_two = stream_two_model(self.in_shape, self.output_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_one = stream_one_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.query_stream_two = stream_two_model(self.kernel_shape, self.kernel_dim, self.cfg, self.device, self.preprocess)
        self.fusion_key = fusion.names[self.fusion_type](input_dim=self.kernel_dim)
        self.fusion_query = fusion.names[self.fusion_type](input_dim=self.kernel_dim)

        print(f"Transport FCN - Stream One: {stream_one_fcn}, Stream Two: {stream_two_fcn}, Stream Fusion: {self.fusion_type}")

    def transport(self, in_tensor, crop, l):
        logits = self.fusion_key(self.key_stream_one(in_tensor), self.key_stream_two(in_tensor, l))
        kernel = self.fusion_query(self.query_stream_one(crop), self.query_stream_two(crop, l))
        return logits, kernel

    def forward(self, inp_img, p, lang_goal, softmax=True):
        """Forward pass."""
        img_unprocessed = np.pad(inp_img, self.padding, mode='constant')        # (384, 224, 6)
        input_data = img_unprocessed
        in_shape = (1,) + input_data.shape
        input_data = input_data.reshape(in_shape)
        in_tensor = torch.from_numpy(input_data).to(dtype=torch.float, device=self.device)      # (1, 384, 224, 6)

        # Rotation pivot.
        pv = np.array([p[0], p[1]]) + self.pad_size     # p = ground truth pixel position for pick object

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size                       # pad_size = 32
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        crop = in_tensor.repeat(self.n_rotations, 1, 1, 1)      # 36 rotations (10 degree increments)
        crop = self.rotator(crop, pivot=pv)
        crop = torch.cat(crop, dim=0)
        crop = crop[:, :, pv[0]-hcrop:pv[0]+hcrop, pv[1]-hcrop:pv[1]+hcrop]     # Take crops around rotated versions of the raw image --> rotated pick object

        logits, kernel = self.transport(in_tensor, crop, lang_goal)

        return self.correlate(logits, kernel, softmax)
    
    
    
class TwoStreamTransportConcatFusionLatNS(TwoStreamTransportLangFusion):
    """Two Stream Transport (a.k.a Place) module with lateral connections"""
    def __init__(self, stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device):

        self.fusion_type = cfg['train']['trans_stream_fusion_type']
        # self.output_dim = 7
        super().__init__(stream_fcn, in_shape, n_rotations, crop_size, preprocess, cfg, device)

    def transport(self, in_tensor, crop, l):
        key_out_one, key_lat_one = self.key_stream_one(in_tensor)
        key_out_two = self.key_stream_two(in_tensor, key_lat_one, l)            # Fuse original image with language goal
        logits = self.fusion_key(key_out_one, key_out_two)                      # (1, 3, 384, 224)

        query_out_one, query_lat_one = self.query_stream_one(crop)
        query_out_two = self.query_stream_two(crop, query_lat_one, l)           # Fuse crop with language goal
        kernel = self.fusion_query(query_out_one, query_out_two)

        return logits, kernel           # (1, 3, 384, 224) and (36, 3, 64, 64)


    def correlate(self, in0, in1, pad_w, pad_h, softmax):
        """Correlate two input tensors."""
        output = F.conv2d(in0, in1, padding=(self.pad_size, self.pad_size))             # Treat in0 and input image and in1 (rotated crops) as conv filters
        output = F.interpolate(output, size=(in0.shape[-2], in0.shape[-1]), mode='bilinear')
        output = output[:,:,self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

        # Extract unpadded image
        w_start = None if pad_w == 0 else (pad_w // 2)
        w_end = None if pad_w - pad_w // 2 == 0 else -(pad_w - pad_w // 2)
        h_start = None if pad_h == 0 else (pad_h // 2)
        h_end = None if pad_h - pad_h // 2 == 0 else -(pad_h - pad_h // 2)
        output = output[:, :, w_start:w_end, h_start:h_end]

        if softmax:
            output_shape = output.shape
            output = output.reshape((1, np.prod(output.shape)))
            output = F.softmax(output, dim=-1)
            output = output.reshape(output_shape[1:])
        return output

    def ns_forward(self, inp_img, prev_filter_map, p, lang_goal, pad_w, pad_h, softmax=True):
        """Forward pass."""
        # Should check scale of RGB and filter_map here (and after clip normalization)
        in_tensor = torch.from_numpy(inp_img).to(dtype=torch.float, device=self.device)
        in_tensor = torch.cat((in_tensor, prev_filter_map), dim=-1)
        in_tensor = torch.nn.functional.pad(in_tensor, tuple(self.padding.flatten()[::-1]), mode='constant')
        in_tensor = in_tensor[None, ...].to(dtype=torch.float, device=self.device)

        # Rotation pivot.
        pv = np.array([p[0] + pad_w // 2, p[1] + pad_h // 2]) + self.pad_size     # p = ground truth pixel position for pick object
        # pv = np.array([p[0], p[1]]) + self.pad_size     # p = ground truth pixel position for pick object

        # Crop before network (default for Transporters CoRL 2020).
        hcrop = self.pad_size                       # pad_size = 32
        in_tensor = in_tensor.permute(0, 3, 1, 2)

        rotated = in_tensor.repeat(self.n_rotations, 1, 1, 1)      # 36 rotations (10 degree increments)
        rotated = self.rotator(rotated, pivot=pv)
        rotated = torch.cat(rotated, dim=0)
        crop = rotated[:, :, pv[0] - hcrop:pv[0] + hcrop, pv[1] - hcrop:pv[1] + hcrop]     # Take crops around rotated versions of the raw image --> rotated pick object

        logits, kernel = self.transport(in_tensor, crop, lang_goal)

        return self.correlate(logits, kernel, pad_w, pad_h, softmax)  # (B, 36, 320, 160) (orig_img: (320, 160, 1))

