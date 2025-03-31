import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models

from train_app.registers.model_registry import model_registry
from train_app.models.base import SemanticSegmentationAdapter

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MultiClassUAFM(nn.Module):

    def rank_algorithm(map): # map needs to be softmaxed
        x = map * torch.log(map + 1e-7)
        x = - 1 * torch.sum(x, dim=1)
        x = torch.exp(x) # max uncertainty = N, min uncertainty = 1
        return x.unsqueeze(1).detach()

    def __init__(self, high_channel, low_channel,out_channel,num_classes, enhance_features=True, uncertain_predict=False):
        super(MultiClassUAFM, self).__init__()
        self.rank = MultiClassUAFM.rank_algorithm
        self.enhance_features = enhance_features
        self.high_channel = high_channel
        self.low_channel = low_channel
        self.out_channel = out_channel
        self.conv_high = BasicConv2d(self.high_channel,self.out_channel,3,1,1)
        self.conv_low = BasicConv2d(self.low_channel,self.out_channel,3,1,1)
        self.conv_fusion = nn.Conv2d(2*self.out_channel,self.out_channel,3,1,1)

        self.seg_out = nn.Conv2d(self.out_channel,num_classes,1)
        self.uncertain_decider = None
        if uncertain_predict:
            self.uncertain_decider = MBDC(in_channel=32, out_channel=num_classes)
        


    def forward(self, feature_low, feature_high, map):
        if self.enhance_features:
            # prepare uncertainty maps of features
            map = torch.softmax(map, dim=1)
            map_upsampled = F.interpolate(map, feature_low.size()[2:], mode='bilinear', align_corners=True)
            
            uncertainty_map_low = self.rank(map_upsampled)
            uncertainty_map_high = self.rank(map)
            
            feature_high = uncertainty_map_high * feature_high
            feature_high = F.interpolate(self.conv_high(feature_high), feature_low.size()[2:], mode='bilinear', align_corners=True)
            feature_low = uncertainty_map_low * feature_low
        else:
            feature_high = self.conv_high(feature_high)
            feature_high = F.interpolate(feature_high, feature_low.size()[2:], mode='bilinear', align_corners=True)

        uncertainty_low = self.conv_low(feature_low)
        seg_fusion = torch.cat((feature_high, uncertainty_low), dim=1)

        seg_fusion = self.conv_fusion(seg_fusion)

        seg = self.seg_out(seg_fusion)

        if self.uncertain_decider is not None:
            decision_uncertainty = self.rank(torch.softmax(seg, dim=1))
            uncertain_seg = self.uncertain_decider(seg_fusion)
            seg = uncertain_seg * decision_uncertainty + seg

        return seg_fusion, seg


class FPN(nn.Module):
    def __init__(self, in_channels,out_channels=256,num_outs=4,
                 start_level=0,
                 end_level=-1,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                1)
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level])for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return tuple(outs)

class SemanticFPNDecoder(nn.Module):
    def __init__(self,channel, feature_strides, num_classes):
        super(SemanticFPNDecoder, self).__init__()
        self.in_channels = [channel, channel, channel, channel]
        self.feature_strides = feature_strides
        self.scale_heads = nn.ModuleList()
        self.channels = channel
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    nn.Sequential(nn.Conv2d(
                        32 if k == 0 else self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=1), nn.BatchNorm2d(self.channels), nn.ReLU(inplace=True)))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.cls_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def forward(self, x):
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            output = output + nn.functional.interpolate(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=False)

        output = self.cls_seg(output)
        return output
    
class CGM(nn.Module):
    def __init__(self, num_classes):
        super(CGM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, num_classes))
        self.prob = nn.Sigmoid()
    
    def forward(self, feature, map):
        features = []
        for i in range(map.shape[1]):
            features.append(self.cgm(feature, map, i).unsqueeze(-1))
        return feature + (torch.cat(features, dim=-1) * self.gamma).sum(-1)
        
    
    def cgm(self, feature, map, cls_idx):
        cls_pred = map[:,cls_idx,:,:].unsqueeze(1)
        m_batchsize, C, height, width = feature.size()
        proj_query = feature.view(m_batchsize, C, -1)
        proj_key = cls_pred.view(m_batchsize, 1, -1).permute(0, 2, 1)
        attention = torch.bmm(proj_query, proj_key)
        attention = attention.unsqueeze(2)
        attention = self.prob(attention)
        out = attention * feature
        return out


class PSM(nn.Module):
    def __init__(self, num_classes):
        super(PSM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, 1, num_classes))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, feature, map):
        features = []
        for i in range(map.shape[1]):
            features.append(self.psm(feature, map, i).unsqueeze(-1))
        features = torch.cat(features, dim=-1)
        weighted_features = features * self.gamma
        return feature + weighted_features.sum(-1)
    

    def psm(self, feature, map, class_id):
        cls_pred = map[:, class_id, :, :].unsqueeze(1)  # Get class-specific prediction map
        m_batchsize, C, height, width = feature.size()
        feature_enhance = []
        step = 4
        for i in range(0,C, step):
            feature_channel = feature[:, i:i+step, :, :].unsqueeze(1)
            proj_query = feature_channel.view(m_batchsize, -1, width * height, 1)
            proj_key = cls_pred.view(m_batchsize, -1, 1, width * height)
            energy = torch.matmul(proj_query, proj_key)
            attention = self.softmax(energy)
            proj_value = feature_channel.view(m_batchsize, -1, 1, width * height)
            out = torch.matmul(proj_value, attention.permute(0, 1, 3, 2)).squeeze(2)
            out = out.view(m_batchsize, step, height, width)
            feature_enhance.append(out)
        return torch.cat(feature_enhance,dim=1)


class MBDC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MBDC, self).__init__()
        self.relu = nn.ReLU(True)
        out_channel_sum = out_channel * 3

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
            BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
            BasicConv2d(out_channel, out_channel, 3, 1, 8, 8),
            BasicConv2d(out_channel, out_channel, 3, 1, 16, 16)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, 1, 0, 1),
            BasicConv2d(out_channel, out_channel, 3, 1, 2, 2),
            BasicConv2d(out_channel, out_channel, 3, 1, 4, 4),
            BasicConv2d(out_channel, out_channel, 3, 1, 8, 8)
        )
        self.conv_cat =BasicConv2d(out_channel_sum, out_channel, 3, 1, 1, 1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1, 1, 0, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

@model_registry.register("MCCMNet")
class MCCMNet(SemanticSegmentationAdapter):
    def __init__(self, channel, num_classes, *args, **kwargs):
        super(MCCMNet, self).__init__(*args, **kwargs)
        vgg16_bn = models.vgg16_bn(pretrained=True)
        self.inc = vgg16_bn.features[:5]
        self.down1 = vgg16_bn.features[5:12]
        self.down2 = vgg16_bn.features[12:22]
        self.down3 = vgg16_bn.features[22:32]
        self.down4 = vgg16_bn.features[32:42]

        self.conv_1 = BasicConv2d(64,channel,3,1,1)
        self.conv_2 = nn.Sequential(MBDC(128,channel))
        self.conv_3 = nn.Sequential(MBDC(256,channel))
        self.conv_4 = nn.Sequential(MBDC(512,channel))
        self.conv_5 = nn.Sequential(MBDC(512,channel))

        self.neck = FPN(in_channels=[channel, channel, channel, channel], out_channels=channel)

        self.decoder = SemanticFPNDecoder(channel = channel,feature_strides=[4, 8, 16, 32],num_classes=num_classes)

        self.cgm = CGM(num_classes)
        self.psm = PSM(num_classes)

        self.ufm_layer4 = MultiClassUAFM(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer3 = MultiClassUAFM(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)
        self.ufm_layer2 = MultiClassUAFM(high_channel = channel,low_channel = channel, out_channel = channel,num_classes=num_classes)
        self.ufm_layer1 = MultiClassUAFM(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes)



    def forward(self, x):
        size = x.size()[2:]
        layer1 = self.inc(x)
        layer2 = self.down1(layer1)
        layer3 = self.down2(layer2)
        layer4 = self.down3(layer3)
        layer5 = self.down4(layer4)


        layer5 = self.conv_5(layer5)
        layer4 = self.conv_4(layer4)
        layer3 = self.conv_3(layer3)
        layer2 = self.conv_2(layer2)
        layer1 = self.conv_1(layer1)

        predict_5 = self.decoder(self.neck([layer2,layer3,layer4,layer5]))

        predict_5_down = F.interpolate(predict_5, layer5.size()[2:], mode='bilinear', align_corners=True)

        layer5 = self.psm(layer5,predict_5_down)
        layer5 = self.cgm(layer5,predict_5_down)

        fusion, predict_4 = self.ufm_layer4(layer4,layer5,predict_5_down)
        fusion, predict_3 = self.ufm_layer3(layer3,fusion,predict_4)
        fusion, predict_2 = self.ufm_layer2(layer2,fusion,predict_3)
        fusion, predict_1 = self.ufm_layer1(layer1,fusion,predict_2)

        return F.interpolate(predict_5, size, mode='bilinear', align_corners=True),\
        F.interpolate(predict_4, size, mode='bilinear', align_corners=True),F.interpolate(predict_3, size, mode='bilinear', align_corners=True),\
               F.interpolate(predict_2, size, mode='bilinear', align_corners=True),F.interpolate(predict_1, size, mode='bilinear', align_corners=True)

