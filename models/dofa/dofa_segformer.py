import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from models.dofa.ops import ConvBnRelu
from models.dofa.dofa_mae import DOFABase16_Weights, dofa_base_patch16_224, dofa_small_patch16_224, dofa_large_patch16_224, dofa_huge_patch16_224


def remove_prefix_from_state_dict(state_dict, prefix):
    # Remove the prefix from the keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # Remove the prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict

class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 feature_strides=[4, 8, 16, 32],
                 num_classes=1, dropout_ratio=0.1):
        super(Decoder, self).__init__()
        embedding_dim = int(in_channels[-1] * 1.5)
        assert len(feature_strides) == len(in_channels)

        self.num_classes = num_classes
        self.in_channels = in_channels
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]).contiguous()
        _c4 = F.interpolate(input=_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]).contiguous()
        _c3 = F.interpolate(input=_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]).contiguous()
        _c2 = F.interpolate(input=_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3]).contiguous()
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x


class DOFASegFormer(nn.Module):
    def __init__(self, encoder, in_channels, classes) -> None:
        super(DOFASegFormer, self).__init__()
        
        self.dofa_encoder = dofa_base_patch16_224(weights=DOFABase16_Weights.DOFA_MAE)
        self.segformer_encoder = smp.encoders.get_encoder(name=encoder, 
                                                          in_channels=in_channels, 
                                                          depth=5, drop_path_rate=0.1)
        # mit_b5_encoder_path = "/gpfs/fs5/nrcan/nrcan_geobase/work/transfer/work/deep_learning/models/multi/RGB/quarte_classes/DG_PS_SP_NB_QC_NFL/maxar_rgb/wv_geoeye_quickbird/pretrained/mit_b5_encoder.pth.tar"
        # segformer_mit_b5_pretrained = torch.load(mit_b5_encoder_path)
        # segformer_mit_b5_pretrained = remove_prefix_from_state_dict(segformer_mit_b5_pretrained, "encoder.")
        # self.segformer_encoder.load_state_dict(segformer_mit_b5_pretrained)
        
        self.segformer_encoder.requires_grad_(False)
        self.dofa_encoder.requires_grad_(False)
        
        mit_b5_in_channels = [64, 128, 320, 512]
        dofa_channel = self.dofa_encoder.fc_norm.normalized_shape[0]
        feature_fusion_channels = []
        for in_channel in mit_b5_in_channels:
            df_sf = dofa_channel + in_channel
            feature_fusion_channels.append(df_sf)
        
        self.c1_feature_fusion = FeatureFusion(in_planes=feature_fusion_channels[0], 
                                               out_planes=768)
        self.c2_feature_fusion = FeatureFusion(in_planes=feature_fusion_channels[1], 
                                               out_planes=768)
        self.c3_feature_fusion = FeatureFusion(in_planes=feature_fusion_channels[2], 
                                               out_planes=768)
        self.c4_feature_fusion = FeatureFusion(in_planes=feature_fusion_channels[3], 
                                               out_planes=768)
        self.decoder = Decoder(in_channels=feature_fusion_channels, num_classes=classes)
        
    def forward(self, img, wavelength):
        img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        x_1 = self.segformer_encoder(img)[2:]
        x_2 = self.dofa_encoder.forward_features(img_resized, wavelength).unsqueeze(2).unsqueeze(3)
        
        c1, c2, c3, c4 = x_1
        c1_x_2_expanded = x_2.expand(-1, -1, c1.shape[2], c1.shape[3])
        c2_x2_expanded = x_2.expand(-1, -1, c2.shape[2], c2.shape[3])
        c3_x2_expanded = x_2.expand(-1, -1, c3.shape[2], c3.shape[3])
        c4_x2_expanded = x_2.expand(-1, -1, c4.shape[2], c4.shape[3])
        
        c1_x_2 = self.c1_feature_fusion(c1, c1_x_2_expanded)
        c2_x_2 = self.c2_feature_fusion(c2, c2_x2_expanded)
        c3_x_2 = self.c3_feature_fusion(c3, c3_x2_expanded)
        c4_x_2 = self.c4_feature_fusion(c4, c4_x2_expanded)
        
        x = self.decoder((c1_x_2, c2_x_2, c3_x_2, c4_x_2))
        x = F.interpolate(input=x, size=img.shape[2:], scale_factor=None, mode='bilinear', align_corners=False)
        return x

if __name__ == '__main__':
    batch_size = 6
    wavelengths = [0.48, 0.56, 0.64]
    img = torch.rand(batch_size, 3, 512, 512)

    model = DOFASegFormer(encoder="mit_b5", in_channels=3, classes=5)
    out = model(img, wavelengths)
    print(f"Output shape: {out.shape}")
