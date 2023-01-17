import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
    The discriminator model.
    """
    def __init__(self, in_channels, c_image=False, classes=1, ndf=64, n_layers=2, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        # if c_image:
        #     in_channels = in_channels + classes
        sequence = [nn.utils.spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=(kw, kw), stride=(2, 2), padding=(padw, padw))), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw), stride=(2, 2), padding=(padw, padw), bias=False)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Dropout(0.5)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kw, kw), stride=(1, 1), padding=(padw, padw), bias=False)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.5)
        ]

        sequence += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=(kw, kw), stride=(1, 1), padding=(padw, padw)))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        # self.fc2 = nn.utils.spectral_norm(nn.Linear(126 * 126, 1))

    def forward(self, y, x=None):
        """
        Forward propagation.
        :param x: high-resolution or super-resolution images
        :param y: groundtruth or segmentation
        :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
        """
        if x is not None:
            # y = torch.cat((y, x), dim=1)
            y = x * y
            # print(f'Discriminator Input: {y.shape}')
        # batch_size = y.size(0)
        output = self.model(y)
        # output = self.fc2(output.view(batch_size, -1))
        # output = self.fc2(output)
        # print(f'Discriminator Output: {output.shape}')
        output = F.sigmoid(output)
        return output

# class ConvolutionalBlock(nn.Module):
#     """
#     A convolutional block, comprising convolutional, BN, activation layers.
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
#         """
#         :param in_channels: number of input channels
#         :param out_channels: number of output channels
#         :param kernel_size: kernel size
#         :param stride: stride
#         :param batch_norm: include a BN layer?
#         :param activation: Type of activation; None if none
#         """
#         super(ConvolutionalBlock, self).__init__()
#
#         if activation is not None:
#             activation = activation.lower()
#             assert activation in {'prelu', 'leakyrelu', 'tanh'}
#
#         # A container that will hold the layers in this convolutional block
#         layers = list()
#
#         # A convolutional layer
#         layers.append(nn.utils.spectral_norm(
#             nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
#                       padding=kernel_size // 2)))
#
#         # A batch normalization (BN) layer, if wanted
#         if batch_norm is True:
#             layers.append(nn.BatchNorm2d(num_features=out_channels))
#
#         # An activation layer, if wanted
#         if activation == 'prelu':
#             layers.append(nn.PReLU())
#         elif activation == 'leakyrelu':
#             layers.append(nn.LeakyReLU(0.2))
#         elif activation == 'tanh':
#             layers.append(nn.Tanh())
#
#         # Put together the convolutional block as a sequence of the layers in this container
#         self.conv_block = nn.Sequential(*layers)
#
#     def forward(self, input):
#         """
#         Forward propagation.
#         :param input: input images, a tensor of size (N, in_channels, w, h)
#         :return: output images, a tensor of size (N, out_channels, w, h)
#         """
#         output = self.conv_block(input)  # (N, out_channels, w, h)
#
#         return output
#
#
# class Discriminator(nn.Module):
#     """
#     The discriminator in the SRGAN, as defined in the paper.
#     """
#
#     def __init__(self, in_channels=3, classes=1, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024, c_image=False):
#         """
#         :param kernel_size: kernel size in all convolutional blocks
#         :param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
#         :param n_blocks: number of convolutional blocks
#         :param fc_size: size of the first fully connected layer
#         :param c_image: set to true if passing image for conditioning
#         """
#         super(Discriminator, self).__init__()
#         # A series of convolutional blocks
#         # The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
#         # The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
#         # The first convolutional block is unique because it does not employ batch normalization
#         conv_blocks = list()
#         if c_image:
#             in_channels = in_channels + classes
#         for i in range(n_blocks):
#             out_channels = (n_channels if i == 0 else in_channels * 2) if i % 2 == 0 else in_channels
#             conv_blocks.append(
#                 ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                    stride=1 if i % 2 == 0 else 2, batch_norm=i != 0, activation='LeakyReLu'))
#             in_channels = out_channels
#         self.conv_blocks = nn.Sequential(*conv_blocks)
#
#         # An adaptive pool layer that resizes it to a standard size
#         # For the default input size of 96 and 8 convolutional blocks, this will have no effect
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
#
#         self.fc1 = nn.utils.spectral_norm(nn.Linear(out_channels * 6 * 6, fc_size))
#
#         self.leaky_relu = nn.LeakyReLU(0.2)
#
#         self.fc2 = nn.utils.spectral_norm(nn.Linear(1024, 1))
#
#         # Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()
#
#     def forward(self, y, x=None):
#         """
#         Forward propagation.
#         :param x: high-resolution or super-resolution images
#         :param y: groundtruth or segmentation
#         :return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
#         """
#         batch_size = y.size(0)
#         if x is not None:
#             y = torch.cat((y, x), dim=1)
#         output = self.conv_blocks(y)
#         output = self.adaptive_pool(output)
#         output = self.fc1(output.view(batch_size, -1))
#         output = self.leaky_relu(output)
#         logit = self.fc2(output)
#         return logit

# create real and fake label
# real_label = torch.full((batch_size,), label_is_real, dtype=images.dtype, device=device)
# fake_label = torch.full((batch_size,), label_is_fake, dtype=images.dtype, device=device)

# Train Discriminator
# for d_parameters in discriminator.parameters():
#     d_parameters.requires_grad = True



# print(f'gt_output_: {gt_output.shape} ground_truth:{ground_truth.shape}')
# sz = tuple(gt_output.size())[2:]
# print(sz)
# ground_truth =functional.interpolate(ground_truth, size=sz, mode='bilinear')
# print(f'gt_output_: {gt_output.shape} ground_truth:{ground_truth.shape}')


# loss_discriminator_gt = criterion_d(gt_output, ground_truth)
# loss_discriminator_gt.backward(retain_graph=True)


# Total Discriminator loss
# loss_discriminator = loss_discriminator_gt + loss_discriminator_sg

# Train Generator
#         for d_parameters in discriminator.parameters():
#             d_parameters.requires_grad = False
