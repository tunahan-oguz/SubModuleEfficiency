from train_app.models.semantic_segmentation.mccmnet import *

@model_registry.register("MCCMNet+")
class MCCMNetPlus(SemanticSegmentationAdapter):
    def __init__(self, channel, num_classes, uncertainty=True, enhance_features=True, uncertain_predict=True, pigm=True, *args, **kwargs):
        super(MCCMNetPlus, self).__init__(*args, **kwargs)
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

        self.cgm = CGM(num_classes) if pigm else lambda x, y: x
        self.psm = PSM(num_classes) if pigm else lambda x, y: x

        self.ufm_layer4 = MultiClassUAFM(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes,
                                          enhance_features=enhance_features, uncertain_predict=uncertain_predict)
        self.ufm_layer3 = MultiClassUAFM(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes,
                                          enhance_features=enhance_features, uncertain_predict=uncertain_predict)
        self.ufm_layer2 = MultiClassUAFM(high_channel = channel,low_channel = channel, out_channel = channel,num_classes=num_classes,
                                          enhance_features=enhance_features, uncertain_predict=uncertain_predict)
        self.ufm_layer1 = MultiClassUAFM(high_channel = channel,low_channel = channel, out_channel = channel, num_classes=num_classes,
                                          enhance_features=enhance_features, uncertain_predict=uncertain_predict)



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

