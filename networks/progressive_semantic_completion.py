from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import config as cfg
from libs.resnet import resnet_v1
from tensorflow.python.ops import init_ops

def res_block(x, channel, name=None):
    res = slim.conv2d(x, channel, [3, 3], scope=name + '_1', padding='SAME')
    res = slim.conv2d(res, channel, [3, 3], scope=name + '_2', padding='SAME', activation_fn=None)
    return tf.nn.relu(tf.add(x, res))

def progressive_mask256(mask0): # mask: occlusion(0)
    # Generate progress mask for facade parsing
    def downscale_mask(mask_fea, kernel_size, name):
        '''down scale operation in U-net'''

        weights_shape = [kernel_size, kernel_size, 1, 1]

        pad_h = kernel_size // 2
        pad_w = kernel_size // 2
        mask_conv = pconv_mask_layer(mask=mask_fea, weights_shape=weights_shape,
                                     pad_size=(pad_h, pad_w),
                                     name=name)
        return mask_conv

    def pconv_mask_layer(mask, weights_shape, pad_size, name):
        ## get shapes
        pad_h, pad_w = pad_size

        weight_for_mask = tf.ones(weights_shape, dtype=tf.float32)

        ### padding

        mask_fea_pad = tf.pad(mask, [[0, 0], [pad_h, pad_h], [pad_h, pad_w], [0, 0]])

        ### convolutions

        stride = 2

        mask_conv = tf.nn.convolution(mask_fea_pad, weight_for_mask, strides=(stride, stride), padding="VALID",
                                      name=name + "_msk_conv")
        # ratio = h*w/mask_conv
        # ratio = tf.where(tf.is_inf(ratio), tf.zeros_like(ratio), ratio) # if value in mask is 0, the division result will be inf
        mask_conv = tf.clip_by_value(mask_conv, 0., 1.)

        return mask_conv

    occ_h, occ_w = mask0.get_shape().as_list()[1:3]
    # 256 x 256
    mask1 = downscale_mask(mask0, 7, 'down1')
    # 128 x 128
    mask2 = downscale_mask(mask1, 5, 'down2')
    # 64 x 64
    mask3 = downscale_mask(mask2, 3, 'down3')
    # 32 x 32
    mask4 = downscale_mask(mask3, 3, 'down4')
    # 16 x 16
    mask5 = downscale_mask(mask4, 3, 'down5')
    # 8 x 8
    mask6 = downscale_mask(mask5, 3, 'down6')
    # 4 x 4
    mask7 = downscale_mask(mask6, 3, 'down7')
    # 2 x 2
    mask8 = downscale_mask(mask7, 3, 'down8')

    occ_h, occ_w = 64, 64
    mask0 = tf.image.resize_nearest_neighbor(mask0, [occ_h, occ_w])
    # 1 x 1

    # mask1 = tf.image.resize_nearest_neighbor(mask1, [occ_h, occ_w])
    # mask2 = tf.image.resize_nearest_neighbor(mask2, [occ_h, occ_w])
    mask3 = tf.image.resize_nearest_neighbor(mask3, [occ_h, occ_w])
    mask4 = tf.image.resize_nearest_neighbor(mask4, [occ_h, occ_w])
    mask5 = tf.image.resize_nearest_neighbor(mask5, [occ_h, occ_w])
    # mask6 = tf.image.resize_nearest_neighbor(mask6, [occ_h, occ_w])
    # mask7 = tf.image.resize_nearest_neighbor(mask7, [occ_h, occ_w])
    mask8 = tf.image.resize_nearest_neighbor(mask8, [occ_h, occ_w])

    mask_collects = [mask0, mask3, mask4, mask5, mask8]

    # progressive masks for learning
    residual_masks = [mask0]
    for i in range(1, len(mask_collects)):
        residual_masks.append((mask_collects[i] - mask_collects[i-1]))

    return mask_collects, residual_masks

def progressive_mask256_prob(mask0): # mask: occlusion(0)
    # Generate progress mask for facade parsing
    def downscale_mask(mask_fea, kernel_size, name):
        '''down scale operation in U-net'''

        weights_shape = [kernel_size, kernel_size, 1, 1]

        pad_h = kernel_size // 2
        pad_w = kernel_size // 2
        mask_conv = pconv_mask_layer(mask=mask_fea, weights_shape=weights_shape,
                                     pad_size=(pad_h, pad_w),
                                     name=name)
        return mask_conv

    def pconv_mask_layer(mask, weights_shape, pad_size, name):
        ## get shapes
        pad_h, pad_w = pad_size

        # weight_for_mask = tf.ones(weights_shape, dtype=tf.float32)

        ### padding

        mask_fea_pad = tf.pad(mask, [[0, 0], [pad_h, pad_h], [pad_h, pad_w], [0, 0]])

        ### convolutions

        stride = 2

        # mask_conv = tf.nn.convolution(mask_fea_pad, weight_for_mask, strides=(stride, stride), padding="VALID",
        #                               name=name + "_msk_conv")
        mask_conv = slim.max_pool2d(mask, [2, 2], scope=name+'_mask_pool')
        # ratio = h*w/mask_conv
        # ratio = tf.where(tf.is_inf(ratio), tf.zeros_like(ratio), ratio) # if value in mask is 0, the division result will be inf
        # mask_conv = tf.clip_by_value(mask_conv, 0., 1.)

        return mask_conv

    occ_h, occ_w = mask0.get_shape().as_list()[1:3]
    # 256 x 256
    mask1 = downscale_mask(mask0, 7, 'down1')
    # 128 x 128
    mask2 = downscale_mask(mask1, 5, 'down2')
    # 64 x 64
    mask3 = downscale_mask(mask2, 3, 'down3')
    # 32 x 32
    mask4 = downscale_mask(mask3, 3, 'down4')
    # 16 x 16
    mask5 = downscale_mask(mask4, 3, 'down5')
    # 8 x 8
    mask6 = downscale_mask(mask5, 3, 'down6')
    # 4 x 4
    mask7 = downscale_mask(mask6, 3, 'down7')
    # 2 x 2
    mask8 = downscale_mask(mask7, 3, 'down8')

    occ_h, occ_w = 64, 64

    # Resize to 64 x 64
    # mask0 = tf.image.resize_nearest_neighbor(mask0, [occ_h, occ_w])
    # mask1 = tf.image.resize_nearest_neighbor(mask1, [occ_h, occ_w])
    # mask2 = tf.image.resize_nearest_neighbor(mask2, [occ_h, occ_w])
    # mask3 = tf.image.resize_nearest_neighbor(mask3, [occ_h, occ_w])
    # mask4 = tf.image.resize_nearest_neighbor(mask4, [occ_h, occ_w])
    # mask5 = tf.image.resize_nearest_neighbor(mask5, [occ_h, occ_w])
    # mask6 = tf.image.resize_nearest_neighbor(mask6, [occ_h, occ_w])
    # mask7 = tf.image.resize_nearest_neighbor(mask7, [occ_h, occ_w])
    # mask8 = tf.image.resize_nearest_neighbor(mask8, [occ_h, occ_w])

    # mask_collects = [mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask8]
    # mask_collects = [mask0, mask2, mask3, mask4, mask5, mask6, mask8]

    # mask_collects = [mask0, mask3, mask8]

    # For ablation study of different steps
    mask_collects = [mask0, mask8]
    # mask_collects = [mask0, mask4, mask8]
    # mask_collects = [mask0, mask3, mask5, mask8]  # 0, 3, 5, 8
    # mask_collects = [mask0, mask3, mask4, mask5, mask8]



    # # progressive masks for learning
    # residual_masks = [mask0]
    # for i in range(1, len(mask_collects)):
    #     residual_masks.append((mask_collects[i] - mask_collects[i-1]))

    residual_masks = mask_collects

    return mask_collects, residual_masks

def progressive_mask256_learning(mask0):
    # Generate progress mask for facade parsing
    def downscale_mask(mask_fea, kernel_size, name):
        '''down scale operation in U-net'''

        weights_shape = [kernel_size, kernel_size, 1, 1]

        pad_h = kernel_size // 2
        pad_w = kernel_size // 2
        mask_conv = pconv_mask_layer(mask=mask_fea, weights_shape=weights_shape,
                                     pad_size=(pad_h, pad_w),
                                     name=name)
        return mask_conv

    def pconv_mask_layer(mask, weights_shape, pad_size, name):
        ## get shapes
        pad_h, pad_w = pad_size

        # weight_for_mask = tf.ones(weights_shape, dtype=tf.float32)

        ### padding

        mask_fea_pad = tf.pad(mask, [[0, 0], [pad_h, pad_h], [pad_h, pad_w], [0, 0]])

        ### convolutions

        stride = 2

        # mask_conv = tf.nn.convolution(mask_fea_pad, weight_for_mask, strides=(stride, stride), padding="VALID",
        #                               name=name + "_msk_conv")
        mask_conv = slim.max_pool2d(mask, [2, 2], scope=name + '_mask_pool')
        # ratio = h*w/mask_conv
        # ratio = tf.where(tf.is_inf(ratio), tf.zeros_like(ratio), ratio) # if value in mask is 0, the division result will be inf
        # mask_conv = tf.clip_by_value(mask_conv, 0., 1.)

        return mask_conv

    occ_h, occ_w = mask0.get_shape().as_list()[1:3]
    # 256 x 256
    mask1 = slim.conv2d(mask0, 8, 7, stride=(2, 2), scope='conv1')#  # downscale_mask(mask0, 7, 'down1')
    # 128 x 128
    mask2 = slim.conv2d(mask1, 16, 5, stride=(2, 2), scope='conv2')# downscale_mask(mask1, 5, 'down2')
    # 64 x 64
    mask3 = slim.conv2d(mask2, 32, 3, rate=2, scope='conv3') # downscale_mask(mask2, 3, 'down3')
    # 64 x 64
    mask4 = slim.conv2d(mask3, 32, 3, rate=4, scope='conv4') # downscale_mask(mask3, 3, 'down4')
    # 64 x 64
    mask5 = slim.conv2d(mask4, 32, 3, rate=8, scope='conv5') # downscale_mask(mask4, 3, 'down5')

    mask1 = slim.conv2d(mask1, 1, 1, scope='logit_mask1', activation_fn=tf.nn.sigmoid)
    mask2 = slim.conv2d(mask2, 1, 1, scope='logit_mask2', activation_fn=tf.nn.sigmoid)
    mask3 = slim.conv2d(mask3, 1, 1, scope='logit_mask3', activation_fn=tf.nn.sigmoid)
    mask4 = slim.conv2d(mask4, 1, 1, scope='logit_mask4', activation_fn=tf.nn.sigmoid)
    mask5 = slim.conv2d(mask5, 1, 1, scope='logit_mask5', activation_fn=tf.nn.sigmoid)

    occ_h, occ_w = 64, 64
    mask0 = tf.image.resize_images(mask0, [occ_h, occ_w])
    # 1 x 1

    mask1 = tf.image.resize_images(mask1, [occ_h, occ_w])
    mask2 = tf.image.resize_images(mask2, [occ_h, occ_w])
    mask3 = tf.image.resize_images(mask3, [occ_h, occ_w])
    mask4 = tf.image.resize_images(mask4, [occ_h, occ_w])
    mask5 = tf.image.resize_images(mask5, [occ_h, occ_w])

    mask_collects = [mask0, mask1, mask2, mask3, mask4, mask5]
    # mask_collects = [mask0, mask1, mask2, mask3, mask4, mask5, mask6, mask8]
    # mask_collects = [mask0, mask2, mask3, mask4, mask5, mask6, mask8]
    # mask_collects = [mask0, mask3, mask4, mask5, mask6, mask8]
    # mask_collects = [mask0, mask3, mask4, mask5, mask8]
    # mask_collects = [mask0, mask3, mask5, mask6, mask8]
    # mask_collects = [mask0, mask2, mask4, mask6, mask8]
    # mask_collects = [mask0, mask3, mask5, mask8]
    # mask_collects = [mask0, mask4, mask8]
    # mask_collects = [mask0, mask8]

    # progressive masks for learning
    residual_masks = [mask0]
    for i in range(1, len(mask_collects)):
        residual_masks.append((mask_collects[i] - mask_collects[i - 1]))

    return mask_collects, residual_masks

# Progressive parsing
def inference_resnet50_progressive(image, is_training=True):

    def atrous_spp16(input_feature, depth=256):
        '''
        ASPP module for deeplabv3+
            if output_stride == 16, rates = [6, 12, 18];
            if output_stride == 8, rate:[12, 24, 36];

        :param input_feature: [b, h, w, c]
        '''
        with tf.variable_scope("aspp"):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=6, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=12, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=18, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            return net
    img_shape = image.get_shape().as_list()

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)

    net = slim.conv2d(net, 256, 1, scope='down_dim')


    with tf.variable_scope("embed", reuse=tf.AUTO_REUSE):
        embed_feat = res_block(net, 256, 'op')
        occ = slim.conv2d(embed_feat, 2, [1, 1], scope='logits_occ', trainable=is_training, activation_fn=None,
                       normalizer_fn=None)
        output_occ = tf.image.resize_images(occ, [img_shape[1], img_shape[2]])

    # occlusion mask
    occ_argmax = tf.expand_dims(tf.argmax(occ, axis=3, name="occ_init_mask"), dim=3)
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_nearest_neighbor(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    mask_collects, residual_masks = progressive_mask256(occ_mask)

    output_collects = []
    # feature of input
    net_shape = net.get_shape().as_list()
    with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
        seg_feat = res_block(net, 256, 'refine')
        seg_cls = seg_feat
        seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                          activation_fn=None,
                          normalizer_fn=None)
        output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
        output_collects.append(output_seg)

    for i in range(1, len(mask_collects)):
        with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
            seg_softmax = tf.nn.softmax(seg)
            seg_softmax = seg_softmax * mask_collects[i-1]
            seg_feat_concat = tf.concat([seg_feat, seg_softmax], axis=3)
            seg_feat_fused = slim.conv2d(seg_feat_concat, 256, 3, scope='fuse1')
            seg_feat_fused = slim.conv2d(seg_feat_fused, 256, 3, scope='fuse2', activation_fn=None)
            seg_feat = tf.nn.relu(seg_feat + seg_feat_fused)

            seg_feat = res_block(seg_feat, 256, 'res1')
            seg_feat = res_block(seg_feat, 256, 'res2')

            seg_feat = atrous_spp16(seg_feat, 256)
            # seg_feat =large_kernel2(seg_feat, 256, 15, 4, 'alk')
            # seg_feat = aspplk(seg_feat, depth=256)
            seg_cls = seg_feat
            seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                              activation_fn=None,
                              normalizer_fn=None)
            output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
            output_collects.append(output_seg)

    # # Combine
    # output_concat = [tf.nn.softmax(output_collects[0])]
    # for i in range(1, len(output_collects)):
    #     output_concat.append(tf.nn.softmax(output_collects[i]))
    # output_concat2 = tf.concat(output_concat, axis=3)
    # output_concat_feat = slim.conv2d(output_concat2, int(cfg.NUM_OF_CLASSESS*len(output_concat)), 3,
    #                                  scope='concat_fuse1')
    # output_concat_feat = slim.conv2d(output_concat_feat, int(cfg.NUM_OF_CLASSESS * len(output_concat)), 3,
    #                                  scope='concat_fuse2')
    # output_seg_concat = slim.conv2d(output_concat_feat, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_concat',
    #                                 trainable=is_training,
    #                                 activation_fn=None,
    #                                 normalizer_fn=None)


    label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    if is_training:
        return label_pred, output_collects, output_occ, embed_feat, mask_collects, residual_masks
    else:
        # output_seg = output_collects[0] * tf.image.resize_nearest_neighbor(residual_masks[0], [img_shape[1], img_shape[2]])
        # for i in range(1, len(output_collects)):
        #     output_seg += output_collects[i] * tf.image.resize_nearest_neighbor(residual_masks[i], [img_shape[1], img_shape[2]])

        return label_pred, output_seg, output_occ, embed_feat, mask_collects, residual_masks

    # # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 512, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #
    # seg_feat, seg = progressive_parsing64(seg_feat, mask_collects, seg)
    #
    # output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    # label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    #
    #
    # return label_pred, output_seg, output_occ, mask_collects, residual_masks


# Progressive parsing with uncertainty map
def inference_resnet50_progressive_uncertainty(image_batch, is_training=True):


    def atrous_spp8(input_feature, depth=256):
        '''
        ASPP module for deeplabv3+
            if output_stride == 16, rates = [6, 12, 18];
            if output_stride == 8, rate:[12, 24, 36];

        :param input_feature: [b, h, w, c]
        '''
        with tf.variable_scope("aspp"):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            return net

    def large_kernel1(x, c, k, r, name):
        '''
        large kernel for facade
        :param x:  input feature
        :param c: output channel
        :param k: kernel size
        :param r: rate for conv
        :return:
        '''
        # 1xk + kx1
        row = slim.conv2d(x, c, [1, k], scope=name + '/row', rate=r)
        col = slim.conv2d(x, c, [k, 1], scope=name + '/col', rate=r)
        y = row + col
        return y

    def CAM(input_feature, depth=256):
        with tf.variable_scope('cam', reuse=tf.AUTO_REUSE):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            net0 = net

            net = large_kernel1(net, 256, 15, 1, 'gcn_a')
            net = net + net0

        return net

    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)

    net = slim.conv2d(net, 256, 1, scope='down_dim')

    # occlusion mask
    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_nearest_neighbor(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.
    # mask_collects, residual_masks = progressive_mask256_learning(occ_mask)  # learning

    # # Change to all one mask
    # for i in range(len(mask_collects)):
    #     mask_collects[i] = mask_collects[-1]

    output_collects = []
    # feature of input
    net_shape = net.get_shape().as_list()
    with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
        seg_feat = res_block(net, 256, 'refine')
        seg_cls = seg_feat
        seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                          activation_fn=None,
                          normalizer_fn=None)
        output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
        output_collects.append(output_seg)

    for i in range(1, len(mask_collects)):
        with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
            seg_softmax = tf.nn.softmax(seg)
            seg_softmax = seg_softmax * mask_collects[i-1]
            seg_feat_concat = tf.concat([seg_feat, seg_softmax], axis=3)
            seg_feat_fused = slim.conv2d(seg_feat_concat, 256, 3, scope='fuse1')
            seg_feat_fused = slim.conv2d(seg_feat_fused, 256, 3, scope='fuse2', activation_fn=None)
            seg_feat = tf.nn.relu(seg_feat + seg_feat_fused)

            seg_feat = res_block(seg_feat, 256, 'res1')
            seg_feat = res_block(seg_feat, 256, 'res2')

            seg_feat = atrous_spp8(seg_feat, 256)
            # seg_feat = CAM(seg_feat, 256)
            # seg_feat =large_kernel2(seg_feat, 256, 15, 4, 'alk')
            # seg_feat = aspplk(seg_feat, depth=256)
            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                # Low level
                low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
                low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
                low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

                # Upsample
                net = tf.image.resize_images(seg_feat, low_level_features_shape)
                net = tf.concat([net, low_level_features], axis=3)
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

            seg_cls = net
            seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                              activation_fn=None,
                              normalizer_fn=None)
            output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
            output_collects.append(output_seg)

            seg = tf.image.resize_images(seg, [img_shape[1] // 8, img_shape[2] // 8])


    # # Combine
    # output_concat = [tf.nn.softmax(output_collects[0])]
    # for i in range(1, len(output_collects)):
    #     output_concat.append(tf.nn.softmax(output_collects[i]))
    # output_concat2 = tf.concat(output_concat, axis=3)
    # output_concat_feat = slim.conv2d(output_concat2, int(cfg.NUM_OF_CLASSESS*len(output_concat)), 3,
    #                                  scope='concat_fuse1')
    # output_concat_feat = slim.conv2d(output_concat_feat, int(cfg.NUM_OF_CLASSESS * len(output_concat)), 3,
    #                                  scope='concat_fuse2')
    # output_seg_concat = slim.conv2d(output_concat_feat, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_concat',
    #                                 trainable=is_training,
    #                                 activation_fn=None,
    #                                 normalizer_fn=None)



    label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    if is_training:
        return label_pred, output_collects, mask_collects, residual_masks
    else:
        # output_seg = output_collects[0] * tf.image.resize_nearest_neighbor(residual_masks[0], [img_shape[1], img_shape[2]])
        # for i in range(1, len(output_collects)):
        #     output_seg += output_collects[i] * tf.image.resize_nearest_neighbor(residual_masks[i], [img_shape[1], img_shape[2]])

        return label_pred, output_collects,  mask_collects, residual_masks

    # # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 512, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #
    # seg_feat, seg = progressive_parsing64(seg_feat, mask_collects, seg)
    #
    # output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    # label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    #
    #
    # return label_pred, output_seg, output_occ, mask_collects, residual_masks

def inference_resnet50_progressive_uncertainty_structure_res(image_batch, is_training=True):


    def atrous_spp8(input_feature, depth=256):
        '''
        ASPP module for deeplabv3+
            if output_stride == 16, rates = [6, 12, 18];
            if output_stride == 8, rate:[12, 24, 36];

        :param input_feature: [b, h, w, c]
        '''
        with tf.variable_scope("aspp"):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            return net

    def large_kernel1(x, c, k, r, name):
        '''
        large kernel for facade
        :param x:  input feature
        :param c: output channel
        :param k: kernel size
        :param r: rate for conv
        :return:
        '''
        # 1xk + kx1
        row = slim.conv2d(x, c, [1, k], scope=name + '/row', rate=r)
        col = slim.conv2d(x, c, [k, 1], scope=name + '/col', rate=r)
        y = row + col
        return y

    def CAM(input_feature, depth=256):
        with tf.variable_scope('cam', reuse=tf.AUTO_REUSE):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            net0 = net

            net = large_kernel1(net, 256, 15, 1, 'gcn_a')
            net = net + net0

        return net

    def calAtt(att_maps):
        ''' Calculate the attmaps of each steps '''
        out = att_maps[0]
        for i in range(1, len(att_maps)):
            out += att_maps[i]
        out = out / tf.reduce_max(out)
        return out

    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)

    net = slim.conv2d(net, 256, 1, scope='down_dim')

    # occlusion mask
    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_nearest_neighbor(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.
    # mask_collects, residual_masks = progressive_mask256_learning(occ_mask)  # learning

    # # Change to all one mask
    # for i in range(len(mask_collects)):
    #     mask_collects[i] = mask_collects[-1]

    output_collects = []
    att_collects = []
    w_att = []
    # feature of input
    net_shape = net.get_shape().as_list()
    with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
        main_feat = net # res_block(net, 256, 'refine')
        seg_feat = slim.conv2d(main_feat, 256, 3, scope='branch')

        seg_feat = atrous_spp8(seg_feat, 256)

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            # Low level
            low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
            low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
            low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

            # Upsample
            net = tf.image.resize_images(seg_feat, low_level_features_shape)
            net = tf.concat([net, low_level_features], axis=3)
            net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

            # sigmoid attention
            uncer_att = slim.conv2d(net, 1, [3, 3], scope='conv_sig_att', activation_fn=None)
            sig_att = tf.nn.sigmoid(uncer_att)
            w_att.append(sig_att)
            cur_att = calAtt(w_att)
            net = net * cur_att
            att_collects.append(tf.image.resize_images(uncer_att, [img_shape[1], img_shape[2]]))

        seg_cls = net
        seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                          activation_fn=None,
                          normalizer_fn=None)
        output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
        output_collects.append(output_seg)
        # seg = tf.image.resize_images(seg, [img_shape[1] // 8, img_shape[2] // 8])
        main_feat = main_feat + seg_feat

    for i in range(1, len(mask_collects)):
        with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
            seg_feat = slim.conv2d(main_feat, 256, 3, scope='branch')
            # # Add Stru
            # seg_softmax = tf.nn.softmax(seg)
            # seg_softmax = seg_softmax * tf.image.resize_images(mask_collects[i-1], seg_softmax.get_shape().as_list()[1:3])
            # seg_feat_concat = tf.concat([seg_feat, seg_softmax], axis=3)
            # seg_feat = slim.conv2d(seg_feat_concat, 256, 3, scope='fuse_concat')

            seg_feat = atrous_spp8(seg_feat, 256)

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                # Low level
                low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
                low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
                low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

                # Upsample
                net = tf.image.resize_images(seg_feat, low_level_features_shape)
                net = tf.concat([net, low_level_features], axis=3)
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

                # sigmoid attention
                uncer_att = slim.conv2d(net, 1, [3, 3], scope='conv_sig_att', activation_fn=None)
                sig_att = tf.nn.sigmoid(uncer_att)
                w_att.append(sig_att)
                cur_att = calAtt(w_att)
                net = net * cur_att
                att_collects.append(tf.image.resize_images(uncer_att, [img_shape[1], img_shape[2]]))

            seg_cls = net
            seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                              activation_fn=None,
                              normalizer_fn=None)
            output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
            output_collects.append(output_seg)
            # seg = tf.image.resize_images(seg, [img_shape[1] // 8, img_shape[2] // 8])
            main_feat = main_feat + seg_feat

    # # Combine
    # output_concat = [tf.nn.softmax(output_collects[0])]
    # for i in range(1, len(output_collects)):
    #     output_concat.append(tf.nn.softmax(output_collects[i]))
    # output_concat2 = tf.concat(output_concat, axis=3)
    # output_concat_feat = slim.conv2d(output_concat2, int(cfg.NUM_OF_CLASSESS*len(output_concat)), 3,
    #                                  scope='concat_fuse1')
    # output_concat_feat = slim.conv2d(output_concat_feat, int(cfg.NUM_OF_CLASSESS * len(output_concat)), 3,
    #                                  scope='concat_fuse2')
    # output_seg_concat = slim.conv2d(output_concat_feat, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_concat',
    #                                 trainable=is_training,
    #                                 activation_fn=None,
    #                                 normalizer_fn=None)

    label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    if is_training:
        return label_pred, output_collects, mask_collects, residual_masks, att_collects
    else:
        # output_seg = output_collects[0] * tf.image.resize_nearest_neighbor(residual_masks[0], [img_shape[1], img_shape[2]])
        # for i in range(1, len(output_collects)):
        #     output_seg += output_collects[i] * tf.image.resize_nearest_neighbor(residual_masks[i], [img_shape[1], img_shape[2]])

        return label_pred, output_collects, mask_collects, residual_masks, att_collects

    # # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 512, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #
    # seg_feat, seg = progressive_parsing64(seg_feat, mask_collects, seg)
    #
    # output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    # label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    #
    #
    # return label_pred, output_seg, output_occ, mask_collects, residual_masks

def inference_resnet50_progressive_facade_parsing_uncertainty(image_batch, is_training=True):

    def atrous_spp8(input_feature, depth=256):
        '''
        ASPP module for deeplabv3+
            if output_stride == 16, rates = [6, 12, 18];
            if output_stride == 8, rate:[12, 24, 36];

        :param input_feature: [b, h, w, c]
        '''
        with tf.variable_scope("aspp"):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            return net

    def large_kernel1(x, c, k, r, name):
        '''
        large kernel for facade
        :param x:  input feature
        :param c: output channel
        :param k: kernel size
        :param r: rate for conv
        :return:
        '''
        # 1xk + kx1
        row = slim.conv2d(x, c, [1, k], scope=name + '/row', rate=r)
        col = slim.conv2d(x, c, [k, 1], scope=name + '/col', rate=r)
        y = row + col
        return y

    # def geometry_module(x, c, name):
    #     '''
    #     The matrix outer product
    #     x: input feature    # b, h, w, c
    #     c: the channel number of output feature
    #     '''
    #
    # def CAM(input_feature, depth=256):
    #     with tf.variable_scope('cam', reuse=tf.AUTO_REUSE):
    #         # 1x1 conv
    #         at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)
    #
    #         # rate = 6
    #         at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)
    #
    #         # rate = 12
    #         at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)
    #
    #         # rate = 18
    #         at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)
    #
    #         # image pooling
    #         img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
    #         img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
    #         img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
    #                                                              input_feature.get_shape().as_list()[2]))
    #
    #         net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
    #                         axis=3, name='atrous_concat')
    #         net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)
    #
    #         net0 = net
    #
    #         net = large_kernel1(net, 256, 15, 1, 'gcn_a')
    #         net = net + net0
    #
    #     return net
    #
    # def sequence_context_fusion_conv1x1(input_feature, depth=256):
    #     with tf.variable_scope('seq_context', reuse=tf.AUTO_REUSE):
    #         step0_conv1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='step0_conv1x1', activation_fn=None)
    #         step0_conv3x3 = slim.conv2d(step0_conv1x1, depth, [3, 3], scope='step0_conv3x3', rate=1, activation_fn=None)
    #         step0_conv3x3 = slim.conv2d(step0_conv3x3, depth, [1, 1], scope='step0_conv1x1_2', activation_fn=None)
    #
    #         step1_conv1x1 = slim.conv2d(step0_conv3x3, depth, [1, 1], scope='step1_conv1x1', activation_fn=None)
    #         step1_conv3x3 = slim.conv2d(step1_conv1x1, depth, [3, 3], scope='step1_conv3x3', rate=12,
    #                                     activation_fn=None)
    #         step1_conv3x3 = slim.conv2d(step1_conv3x3, depth, [1, 1], scope='step1_conv1x1_2', activation_fn=None)
    #
    #         step2_conv1x1 = slim.conv2d(step1_conv3x3, depth, [1, 1], scope='step2_conv1x1', activation_fn=None)
    #         step2_conv3x3 = slim.conv2d(step2_conv1x1, depth, [3, 3], scope='step2_conv3x3', rate=24,
    #                                     activation_fn=None)
    #         step2_conv3x3 = slim.conv2d(step2_conv3x3, depth, [1, 1], scope='step2_conv1x1_2', activation_fn=None)
    #
    #         step3_conv1x1 = slim.conv2d(step2_conv3x3, depth, [1, 1], scope='step3_conv1x1', activation_fn=None)
    #         step3_conv3x3 = slim.conv2d(step3_conv1x1, depth, [3, 3], scope='step3_conv3x3', rate=36,
    #                                     activation_fn=None)
    #         step3_conv3x3 = slim.conv2d(step3_conv3x3, depth, [1, 1], scope='step3_conv1x1_2', activation_fn=None)
    #
    #         # Fusion
    #         fuse3_2 = step3_conv3x3 + step2_conv3x3
    #         fuse3_2 = slim.conv2d(fuse3_2, depth, [1, 1], scope='fuse3_2', activation_fn=None)
    #
    #         fuse2_1 = fuse3_2 + step1_conv3x3
    #         fuse2_1 = slim.conv2d(fuse2_1, depth, [1, 1], scope='fuse2_1', activation_fn=None)
    #
    #         fuse1_0 = fuse2_1 + step0_conv3x3
    #         net = slim.conv2d(fuse1_0, depth, [1, 1], scope='fuse1_0', activation_fn=None)
    #
    #     return net
    #
    # def sequence_context_lk(input_feature, depth=256):
    #     with tf.variable_scope('seq_context_cas', reuse=tf.AUTO_REUSE):
    #         input_feature = slim.conv2d(input_feature, depth, [1, 1], scope='down_dim', activation_fn=None)
    #
    #         lk4 = large_kernel1(input_feature, 256, 15, 4, name='lk4')
    #         lk4 = slim.conv2d(lk4, depth, [1, 1], scope='lk4_refine', activation_fn=None)
    #         lk2 = large_kernel1(input_feature, 256, 15, 2, name='lk2')
    #         lk2 = slim.conv2d(lk2, depth, [1, 1], scope='lk2_refine', activation_fn=None)
    #         lk1 = large_kernel1(input_feature, 256, 15, 1, name='lk1')
    #         lk1 = slim.conv2d(lk1, depth, [1, 1], scope='lk1_refine', activation_fn=None)
    #
    #         fuse1_2 = lk4 + lk2
    #         fuse1_2 = slim.conv2d(fuse1_2, depth, [1, 1], scope='fuse1_2', activation_fn=None)
    #
    #         fuse2_3 = fuse1_2 + lk1
    #         fuse2_3 = slim.conv2d(fuse2_3, depth, [1, 1], scope='fuse2_3', activation_fn=None)
    #
    #         net = fuse2_3
    #
    #     return net
    #
    # def sequence_context_lk_4scale(input_feature, depth=256):
    #     with tf.variable_scope('seq_context_cas', reuse=tf.AUTO_REUSE):
    #         input_feature = slim.conv2d(input_feature, depth, [1, 1], scope='down_dim', activation_fn=None)
    #
    #         lk4 = large_kernel1(input_feature, 256, 15, 4, name='lk4')
    #         lk4 = slim.conv2d(lk4, depth, [1, 1], scope='lk4_refine', activation_fn=None)
    #         lk2 = large_kernel1(input_feature, 256, 15, 2, name='lk2')
    #         lk2 = slim.conv2d(lk2, depth, [1, 1], scope='lk2_refine', activation_fn=None)
    #         lk1 = large_kernel1(input_feature, 256, 15, 1, name='lk1')
    #         lk1 = slim.conv2d(lk1, depth, [1, 1], scope='lk1_refine', activation_fn=None)
    #
    #         fuse1_2 = lk4 + lk2
    #         fuse1_2 = slim.conv2d(fuse1_2, depth, [1, 1], scope='fuse1_2', activation_fn=None)
    #
    #         fuse2_3 = fuse1_2 + lk1
    #         fuse2_3 = slim.conv2d(fuse2_3, depth, [1, 1], scope='fuse2_3', activation_fn=None)
    #
    #         net = fuse2_3 + input_feature
    #
    #     return net
    #
    # def sequence_context_lk_4scale_add(input_feature, depth=256):
    #     with tf.variable_scope('seq_context_cas', reuse=tf.AUTO_REUSE):
    #         input_feature = slim.conv2d(input_feature, depth, [1, 1], scope='down_dim', activation_fn=None)
    #
    #         lk4 = large_kernel1(input_feature, 256, 15, 4, name='lk4')
    #         lk4 = slim.conv2d(lk4, depth, [1, 1], scope='lk4_refine', activation_fn=None)
    #         lk2 = large_kernel1(input_feature, 256, 15, 2, name='lk2')
    #         lk2 = slim.conv2d(lk2, depth, [1, 1], scope='lk2_refine', activation_fn=None)
    #         lk1 = large_kernel1(input_feature, 256, 15, 1, name='lk1')
    #         lk1 = slim.conv2d(lk1, depth, [1, 1], scope='lk1_refine', activation_fn=None)
    #
    #
    #
    #         net = lk4 + lk2 + lk1 + input_feature
    #
    #     return net

    def sequence_context_lk_4scale_concat(input_feature, depth=256):
        with tf.variable_scope('seq_context_cas', reuse=tf.AUTO_REUSE):
            input_feature = slim.conv2d(input_feature, depth, [1, 1], scope='down_dim', activation_fn=None)

            lk4 = large_kernel1(input_feature, 256, 15, 4, name='lk4')
            lk4 = slim.conv2d(lk4, depth, [1, 1], scope='lk4_refine', activation_fn=None)
            lk2 = large_kernel1(input_feature, 256, 15, 2, name='lk2')
            lk2 = slim.conv2d(lk2, depth, [1, 1], scope='lk2_refine', activation_fn=None)
            lk1 = large_kernel1(input_feature, 256, 15, 1, name='lk1')
            lk1 = slim.conv2d(lk1, depth, [1, 1], scope='lk1_refine', activation_fn=None)

            net = tf.concat([lk4, lk2, lk1, input_feature],
                            axis=3, name='concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

        return net

    def sequence_context_lk_3scale(input_feature, depth=256):
        with tf.variable_scope('seq_context_cas', reuse=tf.AUTO_REUSE):
            input_feature = slim.conv2d(input_feature, depth, [1, 1], scope='down_dim', activation_fn=None)

            lk4 = large_kernel1(input_feature, 256, 15, 4, name='lk4')
            lk4 = slim.conv2d(lk4, depth, [1, 1], scope='lk4_refine', activation_fn=None)
            lk2 = large_kernel1(input_feature, 256, 15, 2, name='lk2')
            lk2 = slim.conv2d(lk2, depth, [1, 1], scope='lk2_refine', activation_fn=None)
            lk1 = large_kernel1(input_feature, 256, 15, 1, name='lk1')
            lk1 = slim.conv2d(lk1, depth, [1, 1], scope='lk1_refine', activation_fn=None)

            net = tf.concat([lk4, lk2, lk1],
                            axis=3, name='concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

        return net

    def calAtt(att_maps):
        ''' Calculate the attmaps of each steps '''
        out = att_maps[0]
        for i in range(1, len(att_maps)):
            out += att_maps[i]
        out = out / tf.reduce_max(out)
        return out

    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)

    net = slim.conv2d(net, 256, 1, scope='down_dim')

    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_images(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.    [mask0, mask3, mask5, mask8]  # 0, 3, 5, 8

    output_collects = []
    att_collects = []

    # feature of input
    net_shape = [net.get_shape().as_list()[1], net.get_shape().as_list()[2]]
    feats_collects = []

    main_feat = net
    for i in range(len(mask_collects)):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
                seg_feat = slim.conv2d(main_feat, 256, 3, scope='branch1')

                seg_feat = seg_feat

                seg_feat = sequence_context_lk_4scale_concat(seg_feat, 256)
                # seg_feat = sequence_context_lk_3scale(seg_feat, 256)          # For Clean ECP Dataset only

                seg_feat = slim.conv2d(seg_feat, 256, 3, scope='branch2')

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                # Low level
                low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
                low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
                low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

                # Upsample
                net = tf.image.resize_images(seg_feat, low_level_features_shape)
                net = tf.concat([net, low_level_features], axis=3)
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
                seg_feat2 = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

            seg_cls = seg_feat2
            feats_collects.append(tf.image.resize_images(seg_cls, [img_shape[1], img_shape[2]]))
            seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                              activation_fn=None,
                              normalizer_fn=None)
        output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
        output_collects.append(output_seg)

        main_feat = main_feat + seg_feat

    label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    if is_training:
        return label_pred, output_collects, mask_collects, residual_masks, att_collects, feats_collects
    else:
        # output_seg = output_collects[0] * tf.image.resize_nearest_neighbor(residual_masks[0], [img_shape[1], img_shape[2]])
        # for i in range(1, len(output_collects)):
        #     output_seg += output_collects[i] * tf.image.resize_nearest_neighbor(residual_masks[i], [img_shape[1], img_shape[2]])
        return label_pred, output_collects, mask_collects, residual_masks, att_collects

    # # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 512, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #
    # seg_feat, seg = progressive_parsing64(seg_feat, mask_collects, seg)
    #
    # output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    # label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    #
    #
    # return label_pred, output_seg, output_occ, mask_collects, residual_masks



def inference_resnet50_progressive_uncertainty_attention(image_batch, is_training=True):


    def atrous_spp8(input_feature, depth=256):
        '''
        ASPP module for deeplabv3+
            if output_stride == 16, rates = [6, 12, 18];
            if output_stride == 8, rate:[12, 24, 36];

        :param input_feature: [b, h, w, c]
        '''
        with tf.variable_scope("aspp"):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            return net

    def large_kernel1(x, c, k, r, name):
        '''
        large kernel for facade
        :param x:  input feature
        :param c: output channel
        :param k: kernel size
        :param r: rate for conv
        :return:
        '''
        # 1xk + kx1
        row = slim.conv2d(x, c, [1, k], scope=name + '/row', rate=r)
        col = slim.conv2d(x, c, [k, 1], scope=name + '/col', rate=r)
        y = row + col
        return y

    def CAM(input_feature, depth=256):
        with tf.variable_scope('cam', reuse=tf.AUTO_REUSE):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            net0 = net

            net = large_kernel1(net, 256, 15, 1, 'gcn_a')
            net = net + net0

        return net

    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)

    net = slim.conv2d(net, 256, 1, scope='down_dim')

    # occlusion mask
    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_nearest_neighbor(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.
    # mask_collects, residual_masks = progressive_mask256_learning(occ_mask)  # learning

    # # Change to all one mask
    # for i in range(len(mask_collects)):
    #     mask_collects[i] = mask_collects[-1]

    # seg_select_collects = []
    seg_collects = []

    # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 256, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #     output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    #     output_collects.append(output_seg)

    # att_select_collects = []
    # att_update_collects = []
    first_un_prob = mask_collects[0]

    seg_feat = net
    for i in range(1, len(mask_collects)):
        with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
            uncer_prob = mask_collects[i - 1]
            seg_feat_concat = tf.concat([seg_feat, uncer_prob, first_un_prob], axis=3)
            seg_feat_fused = slim.conv2d(seg_feat_concat, 256, 3, scope='select_fuse1')
            seg_feat = slim.conv2d(seg_feat_fused, 256, 3, scope='select_fuse2', activation_fn=tf.nn.sigmoid)

            # seg_feat = atrous_spp8(seg_feat, 256)
            # seg_feat = CAM(seg_feat, 256)
            # seg_feat =large_kernel2(seg_feat, 256, 15, 4, 'alk')
            # seg_feat = aspplk(seg_feat, depth=256)

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                # Low level
                low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
                low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
                low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

                # Upsample
                net = tf.image.resize_images(seg_feat, low_level_features_shape)
                net = tf.concat([net, low_level_features], axis=3)
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

                seg_cls = net
                seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                                  activation_fn=None,
                                  normalizer_fn=None)
                output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
                seg_collects.append(output_seg)

    # with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
    #             # Low level
    #             low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
    #             low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
    #             low_level_features_shape = low_level_features.get_shape().as_list()[1:3]
    #
    #             # Upsample
    #             net = tf.image.resize_images(seg_feat, low_level_features_shape)
    #             net = tf.concat([net, low_level_features], axis=3)
    #             net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
    #             net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')
    #
    #             seg_cls = net
    #             seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                               activation_fn=None,
    #                               normalizer_fn=None)
    #             output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
            # seg = tf.image.resize_images(seg, [img_shape[1] // 8, img_shape[2] // 8])


    # # Combine
    # output_concat = [tf.nn.softmax(output_collects[0])]
    # for i in range(1, len(output_collects)):
    #     output_concat.append(tf.nn.softmax(output_collects[i]))
    # output_concat2 = tf.concat(output_concat, axis=3)
    # output_concat_feat = slim.conv2d(output_concat2, int(cfg.NUM_OF_CLASSESS*len(output_concat)), 3,
    #                                  scope='concat_fuse1')
    # output_concat_feat = slim.conv2d(output_concat_feat, int(cfg.NUM_OF_CLASSESS * len(output_concat)), 3,
    #                                  scope='concat_fuse2')
    # output_seg_concat = slim.conv2d(output_concat_feat, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_concat',
    #                                 trainable=is_training,
    #                                 activation_fn=None,
    #                                 normalizer_fn=None)



    label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    if is_training:
        return label_pred, seg_collects, mask_collects, residual_masks
    else:
        # output_seg = output_collects[0] * tf.image.resize_nearest_neighbor(residual_masks[0], [img_shape[1], img_shape[2]])
        # for i in range(1, len(output_collects)):
        #     output_seg += output_collects[i] * tf.image.resize_nearest_neighbor(residual_masks[i], [img_shape[1], img_shape[2]])

        return label_pred, seg_collects, mask_collects, residual_masks

    # # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 512, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #
    # seg_feat, seg = progressive_parsing64(seg_feat, mask_collects, seg)
    #
    # output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    # label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    #
    #
    # return label_pred, output_seg, output_occ, mask_collects, residual_masks

def inference_resnet50_progressive_uncertainty_loss(image_batch, is_training=True):


    def atrous_spp8(input_feature, depth=256):
        '''
        ASPP module for deeplabv3+
            if output_stride == 16, rates = [6, 12, 18];
            if output_stride == 8, rate:[12, 24, 36];

        :param input_feature: [b, h, w, c]
        '''
        with tf.variable_scope("aspp"):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            return net

    def large_kernel1(x, c, k, r, name):
        '''
        large kernel for facade
        :param x:  input feature
        :param c: output channel
        :param k: kernel size
        :param r: rate for conv
        :return:
        '''
        # 1xk + kx1
        row = slim.conv2d(x, c, [1, k], scope=name + '/row', rate=r)
        col = slim.conv2d(x, c, [k, 1], scope=name + '/col', rate=r)
        y = row + col
        return y

    def CAM(input_feature, depth=256):
        with tf.variable_scope('cam', reuse=tf.AUTO_REUSE):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            net0 = net

            net = large_kernel1(net, 256, 15, 1, 'gcn_a')
            net = net + net0

        return net

    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)

    net = slim.conv2d(net, 256, 1, scope='down_dim')

    # occlusion mask
    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_nearest_neighbor(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.
    # mask_collects, residual_masks = progressive_mask256_learning(occ_mask)  # learning

    # # Change to all one mask
    # for i in range(len(mask_collects)):
    #     mask_collects[i] = mask_collects[-1]

    # seg_select_collects = []
    seg_collects = []

    # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 256, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #     output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    #     output_collects.append(output_seg)

    # att_select_collects = []
    # att_update_collects = []
    first_un_prob = mask_collects[0]
    zeros = tf.zeros(first_un_prob.shape)

    main_feat = net
    for i in range(len(mask_collects)):
        with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
            seg_feat = main_feat
            if i == 0:
                last_uncer = zeros
            else:
                last_uncer = mask_collects[i-1]
            seg_feat_concat = tf.concat([seg_feat, last_uncer], axis=3)
            seg_feat = slim.conv2d(seg_feat_concat, 256, 3, scope='select_fuse1')
            seg_feat = slim.conv2d(seg_feat, 256, [3, 3], scope='seg_feat_1')
            seg_feat = atrous_spp8(seg_feat, 256)
            # seg_feat = atrous_large_kernel(seg_feat, 256, 64, 1, 'alk_row_col')  # 64x64

            # seg_feat = CAM(seg_feat, 256)
            # seg_feat =large_kernel2(seg_feat, 256, 15, 4, 'alk')
            # seg_feat = aspplk(seg_feat, depth=256)

            with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
                # Low level
                low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
                low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
                low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

                # Upsample
                seg_cls = tf.image.resize_images(seg_feat, low_level_features_shape)
                seg_cls = tf.concat([seg_cls, low_level_features], axis=3)
                seg_cls = slim.conv2d(seg_cls, 256, [3, 3], scope='conv_3x3_1')
                seg_cls = slim.conv2d(seg_cls, 256, [3, 3], scope='conv_3x3_2')


                seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                                  activation_fn=None,
                                  normalizer_fn=None)
                output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
                seg_collects.append(output_seg)
            main_feat = main_feat + seg_feat

    # with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
    #             # Low level
    #             low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
    #             low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
    #             low_level_features_shape = low_level_features.get_shape().as_list()[1:3]
    #
    #             # Upsample
    #             net = tf.image.resize_images(seg_feat, low_level_features_shape)
    #             net = tf.concat([net, low_level_features], axis=3)
    #             net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
    #             net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')
    #
    #             seg_cls = net
    #             seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                               activation_fn=None,
    #                               normalizer_fn=None)
    #             output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
            # seg = tf.image.resize_images(seg, [img_shape[1] // 8, img_shape[2] // 8])


    # # Combine
    # output_concat = [tf.nn.softmax(output_collects[0])]
    # for i in range(1, len(output_collects)):
    #     output_concat.append(tf.nn.softmax(output_collects[i]))
    # output_concat2 = tf.concat(output_concat, axis=3)
    # output_concat_feat = slim.conv2d(output_concat2, int(cfg.NUM_OF_CLASSESS*len(output_concat)), 3,
    #                                  scope='concat_fuse1')
    # output_concat_feat = slim.conv2d(output_concat_feat, int(cfg.NUM_OF_CLASSESS * len(output_concat)), 3,
    #                                  scope='concat_fuse2')
    # output_seg_concat = slim.conv2d(output_concat_feat, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_concat',
    #                                 trainable=is_training,
    #                                 activation_fn=None,
    #                                 normalizer_fn=None)



    label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    if is_training:
        return label_pred, seg_collects, mask_collects, residual_masks
    else:
        # output_seg = output_collects[0] * tf.image.resize_nearest_neighbor(residual_masks[0], [img_shape[1], img_shape[2]])
        # for i in range(1, len(output_collects)):
        #     output_seg += output_collects[i] * tf.image.resize_nearest_neighbor(residual_masks[i], [img_shape[1], img_shape[2]])

        return label_pred, seg_collects, mask_collects, residual_masks

    # # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 512, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #
    # seg_feat, seg = progressive_parsing64(seg_feat, mask_collects, seg)
    #
    # output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    # label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    #
    #
    # return label_pred, output_seg, output_occ, mask_collects, residual_masks

def inference_resnet50_uncertainty_direct_fusion(image_batch, is_training=True):


    def atrous_spp8(input_feature, depth=256):
        '''
        ASPP module for deeplabv3+
            if output_stride == 16, rates = [6, 12, 18];
            if output_stride == 8, rate:[12, 24, 36];

        :param input_feature: [b, h, w, c]
        '''
        with tf.variable_scope("aspp"):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            return net

    def large_kernel1(x, c, k, r, name):
        '''
        large kernel for facade
        :param x:  input feature
        :param c: output channel
        :param k: kernel size
        :param r: rate for conv
        :return:
        '''
        # 1xk + kx1
        row = slim.conv2d(x, c, [1, k], scope=name + '/row', rate=r)
        col = slim.conv2d(x, c, [k, 1], scope=name + '/col', rate=r)
        y = row + col
        return y

    def CAM(input_feature, depth=256):
        with tf.variable_scope('cam', reuse=tf.AUTO_REUSE):
            # 1x1 conv
            at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

            # rate = 6
            at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

            # rate = 12
            at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

            # rate = 18
            at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

            # image pooling
            img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
            img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
            img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                                 input_feature.get_shape().as_list()[2]))

            net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                            axis=3, name='atrous_concat')
            net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

            net0 = net

            net = large_kernel1(net, 256, 15, 1, 'gcn_a')
            net = net + net0

        return net

    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)

    net = slim.conv2d(net, 256, 1, scope='down_dim')

    # occlusion mask
    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_images(occ_mask, [img_shape[1] // 8, img_shape[2] // 8])  # 0:occ

    with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
        seg_feat_concat = tf.concat([net, occ_mask], axis=3)
        seg_feat_fused = slim.conv2d(seg_feat_concat, 256, 3, scope='fuse1')
        seg_feat_fused = slim.conv2d(seg_feat_fused, 256, 3, scope='fuse2', activation_fn=None)

        seg_feat = atrous_spp8(seg_feat_fused, 256)

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            # Low level
            low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
            low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
            low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

            # Upsample
            net = tf.image.resize_images(seg_feat, low_level_features_shape)
            net = tf.concat([net, low_level_features], axis=3)
            net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')

        seg_cls = net
        logits = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
                          activation_fn=None,
                          normalizer_fn=None)
        logits = tf.image.resize_images(logits, [img_shape[1], img_shape[2]])
        output_seg = tf.image.resize_images(logits, [img_shape[1], img_shape[2]])

    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    # mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.
    # mask_collects, residual_masks = progressive_mask256_learning(occ_mask)  # learning

    # # Change to all one mask
    # for i in range(len(mask_collects)):
    #     mask_collects[i] = mask_collects[-1]

    # output_collects = []
    # # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 256, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #     output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    #     output_collects.append(output_seg)

    # for i in range(1, len(mask_collects)):
    #     with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #         seg_softmax = tf.nn.softmax(seg)
    #         seg_softmax = seg_softmax * mask_collects[i-1]
    #         seg_feat_concat = tf.concat([seg_feat, seg_softmax], axis=3)
    #         seg_feat_fused = slim.conv2d(seg_feat_concat, 256, 3, scope='fuse1')
    #         seg_feat_fused = slim.conv2d(seg_feat_fused, 256, 3, scope='fuse2', activation_fn=None)
    #         seg_feat = tf.nn.relu(seg_feat + seg_feat_fused)
    #
    #         seg_feat = res_block(seg_feat, 256, 'res1')
    #         seg_feat = res_block(seg_feat, 256, 'res2')
    #
    #         seg_feat = atrous_spp8(seg_feat, 256)
    #         # seg_feat = CAM(seg_feat, 256)
    #         # seg_feat =large_kernel2(seg_feat, 256, 15, 4, 'alk')
    #         # seg_feat = aspplk(seg_feat, depth=256)
    #         with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
    #             # Low level
    #             low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
    #             low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
    #             low_level_features_shape = low_level_features.get_shape().as_list()[1:3]
    #
    #             # Upsample
    #             net = tf.image.resize_images(seg_feat, low_level_features_shape)
    #             net = tf.concat([net, low_level_features], axis=3)
    #             net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
    #             net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')
    #
    #         seg_cls = net
    #         seg = slim.conv2d(seg_cls, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                           activation_fn=None,
    #                           normalizer_fn=None)
    #         output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    #         output_collects.append(output_seg)
    #
    #         seg = tf.image.resize_images(seg, [img_shape[1] // 8, img_shape[2] // 8])


    # # Combine
    # output_concat = [tf.nn.softmax(output_collects[0])]
    # for i in range(1, len(output_collects)):
    #     output_concat.append(tf.nn.softmax(output_collects[i]))
    # output_concat2 = tf.concat(output_concat, axis=3)
    # output_concat_feat = slim.conv2d(output_concat2, int(cfg.NUM_OF_CLASSESS*len(output_concat)), 3,
    #                                  scope='concat_fuse1')
    # output_concat_feat = slim.conv2d(output_concat_feat, int(cfg.NUM_OF_CLASSESS * len(output_concat)), 3,
    #                                  scope='concat_fuse2')
    # output_seg_concat = slim.conv2d(output_concat_feat, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits_concat',
    #                                 trainable=is_training,
    #                                 activation_fn=None,
    #                                 normalizer_fn=None)



    label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)

    return label_pred, logits

    # if is_training:
    #     return label_pred, output_collects, mask_collects, residual_masks
    # else:
    #     # output_seg = output_collects[0] * tf.image.resize_nearest_neighbor(residual_masks[0], [img_shape[1], img_shape[2]])
    #     # for i in range(1, len(output_collects)):
    #     #     output_seg += output_collects[i] * tf.image.resize_nearest_neighbor(residual_masks[i], [img_shape[1], img_shape[2]])
    #
    #     return label_pred, output_collects,  mask_collects, residual_masks

    # # feature of input
    # net_shape = net.get_shape().as_list()
    # with tf.variable_scope("seg", reuse=tf.AUTO_REUSE):
    #     seg_feat = res_block(net, 512, 'refine')
    #     seg_cls = seg_feat
    #     seg = slim.conv2d(seg_cls, cfg.NUM_OF_CLASSESS, [1, 1], scope='logits', trainable=is_training,
    #                       activation_fn=None,
    #                       normalizer_fn=None)
    #
    # seg_feat, seg = progressive_parsing64(seg_feat, mask_collects, seg)
    #
    # output_seg = tf.image.resize_images(seg, [img_shape[1], img_shape[2]])
    # label_pred = tf.expand_dims(tf.argmax(output_seg, axis=3, name="prediction"), dim=3)
    #
    #
    # return label_pred, output_seg, output_occ, mask_collects, residual_masks


def atrous_spp16(input_feature, depth=256):
    '''
    ASPP module for deeplabv3+
        if output_stride == 16, rates = [6, 12, 18];
        if output_stride == 8, rate:[12, 24, 36];

    :param input_feature: [b, h, w, c]
    '''
    with tf.variable_scope("aspp"):
        # 1x1 conv
        at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

        # rate = 6
        at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=6, activation_fn=None)

        # rate = 12
        at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=12, activation_fn=None)

        # rate = 18
        at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=18, activation_fn=None)

        # image pooling
        img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
        img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
        img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                             input_feature.get_shape().as_list()[2]))

        net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
                        axis=3, name='atrous_concat')
        net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

        return net

def guidance_fusion(c, p, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        out_channel = c.get_shape().as_list()[3]
        cp_concat = tf.concat([c, p], axis=-1, name='concat')

        cp_conv = slim.conv2d(cp_concat, out_channel, 3, scope='fusion_conv1')
        cp_conv = slim.conv2d(cp_conv, out_channel, 3, scope='fusion_conv2')

        cp_conv = atrous_spp16(cp_conv)
        return cp_conv

def inference_resnet50_fpn0(image_batch, is_training=True):
    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # occlusion mask
    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_nearest_neighbor(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=16,
                                                 spatial_squeeze=False)
        # residural layer: 3 4 6 3
        c2 = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']     # 256
        c3 = end_points['resnet_v1_50/block2/unit_3/bottleneck_v1']     # 128
        c4 = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']     # 64
        c5 = end_points['resnet_v1_50/block4']                          # 32

        # c4 = slim.conv2d(c4, 256, 1, scope='c4_lateral_layer')
        # c3 = slim.conv2d(c3, 256, 1, scope='c3_lateral_layer')
        # c2 = slim.conv2d(c2, 256, 1, scope='c2_lateral_layer')

        p5 = atrous_spp16(c5)
        y5 = slim.conv2d(p5, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p5', activation_fn=None, normalizer_fn=None)
        logit5 = tf.image.resize_images(y5, [img_shape[1], img_shape[2]])

        # p5_4 = tf.image.resize_images(p5, c4.get_shape().as_list()[1: 3])
        # p4 = guidance_fusion(c4, p5_4, 'gf4')
        # y4 = slim.conv2d(p4, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p4', activation_fn=None, normalizer_fn=None)
        # logit4 = tf.image.resize_images(y4, [img_shape[1], img_shape[2]])
        #
        # p4_3 = tf.image.resize_images(p4, c3.get_shape().as_list()[1: 3])
        # p3 = guidance_fusion(c3, p4_3, 'gf3')
        # y3 = slim.conv2d(p3, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p3', activation_fn=None, normalizer_fn=None)
        # logit3 = tf.image.resize_images(y3, [img_shape[1], img_shape[2]])
        #
        # p3_2 = tf.image.resize_images(p3, c2.get_shape().as_list()[1: 3])
        # p2 = guidance_fusion(c2, p3_2, 'gf2')
        # y2 = slim.conv2d(p2, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p2', activation_fn=None, normalizer_fn=None)
        # logit2 = tf.image.resize_images(y2, [img_shape[1], img_shape[2]])
        #
        # y_sum = tf.concat([logit5, logit4, logit3, logit2], axis=3)
        # logit1 = slim.conv2d(y_sum, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p1', activation_fn=None, normalizer_fn=None)
        #

        label_pred = tf.expand_dims(tf.argmax(logit5, axis=3, name="prediction"), dim=3)
        logits = [logit5] # [logit2, logit3, logit4, logit5, logit1]

        return label_pred, logits, mask_collects, residual_masks

def inference_resnet50_fpn(image_batch, is_training=True):
    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # occlusion mask
    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_nearest_neighbor(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=32,
                                                 spatial_squeeze=False)
        # residural layer: 3 4 6 3
        c2 = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']     # 256
        c3 = end_points['resnet_v1_50/block2/unit_3/bottleneck_v1']     # 128
        c4 = end_points['resnet_v1_50/block3/unit_5/bottleneck_v1']     # 64
        c5 = end_points['resnet_v1_50/block4']                          # 32

        c4 = slim.conv2d(c4, 256, 1, scope='c4_lateral_layer')
        c3 = slim.conv2d(c3, 256, 1, scope='c3_lateral_layer')
        c2 = slim.conv2d(c2, 256, 1, scope='c2_lateral_layer')

        p5 = atrous_spp16(c5)
        y5 = slim.conv2d(p5, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p5', activation_fn=None, normalizer_fn=None)
        logit5 = tf.image.resize_images(y5, [img_shape[1], img_shape[2]])

        p5_4 = tf.image.resize_images(p5, c4.get_shape().as_list()[1: 3])
        p4 = guidance_fusion(c4, p5_4, 'gf')
        y4 = slim.conv2d(p4, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p4', activation_fn=None, normalizer_fn=None)
        logit4 = tf.image.resize_images(y4, [img_shape[1], img_shape[2]])

        p4_3 = tf.image.resize_images(p4, c3.get_shape().as_list()[1: 3])
        p3 = guidance_fusion(c3, p4_3, 'gf')
        y3 = slim.conv2d(p3, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p3', activation_fn=None, normalizer_fn=None)
        logit3 = tf.image.resize_images(y3, [img_shape[1], img_shape[2]])

        p3_2 = tf.image.resize_images(p3, c2.get_shape().as_list()[1: 3])
        p2 = guidance_fusion(c2, p3_2, 'gf')
        y2 = slim.conv2d(p2, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p2', activation_fn=None, normalizer_fn=None)
        logit2 = tf.image.resize_images(y2, [img_shape[1], img_shape[2]])

        feat2_shape = p2.get_shape().as_list()
        p_sum = tf.image.resize_images(p5, feat2_shape[1: 3]) + \
            tf.image.resize_images(p4, feat2_shape[1: 3]) + \
            tf.image.resize_images(p3, feat2_shape[1: 3]) + p2
        p_sum = slim.conv2d(p_sum, 256, 3, scope='sum_conv')
        y_sum = slim.conv2d(p_sum, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_p1', activation_fn=None, normalizer_fn=None)
        logit1 = tf.image.resize_images(y_sum, [img_shape[1], img_shape[2]])

        label_pred = tf.expand_dims(tf.argmax(logit1, axis=3, name="prediction"), dim=3)
        logits = [logit5, logit4, logit3, logit2, logit1]

        return label_pred, logits, mask_collects, residual_masks


def inference_resnet50_backbone(image_batch, is_training=True):
    image = image_batch[:, :, :, 0:3]
    uncertainty = image_batch[:, :, :, 3:4]
    img_shape = image.get_shape().as_list()

    # occlusion mask
    occ_argmax = uncertainty
    occ_mask = tf.cast(occ_argmax, tf.float32)  # 0: occ
    occ_mask = tf.image.resize_nearest_neighbor(occ_mask, [img_shape[1] // 2, img_shape[2] // 2])  # 0:occ
    # mask_collects, residual_masks = progressive_mask256(occ_mask)   # binary
    mask_collects, residual_masks = progressive_mask256_prob(occ_mask)   # prob.

    # Large scale
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        # Feature extractor: ResNet50
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net_l, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                     global_pool=False, output_stride=8,
                                                     spatial_squeeze=False)
        # residural layer: 3 4 6 3
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            net_l = atrous_spp16(net_l)
            # Low level
            low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
            low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
            low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

            # Upsample
            net_l = tf.image.resize_images(net_l, low_level_features_shape)
            net_l = tf.concat([net_l, low_level_features], axis=3)
            net_l = slim.conv2d(net_l, 256, [3, 3], scope='conv_3x3_1')
            net_l = slim.conv2d(net_l, 256, [3, 3], scope='conv_3x3_2')

            logit_l = slim.conv2d(net_l, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', activation_fn=None,
                                  normalizer_fn=None)
            logit_l_resize = tf.image.resize_images(logit_l, [img_shape[1], img_shape[2]])

    # Small scale
    image_g = tf.image.resize_images(image, [img_shape[1] // 2, img_shape[2] // 2])
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        # Feature extractor: ResNet50
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net_g, end_points = resnet_v1.resnet_v1_50(image_g, num_classes=None, is_training=None,
                                                     global_pool=False, output_stride=8,
                                                     spatial_squeeze=False)

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            net_g = atrous_spp16(net_g)
            # Low level
            low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
            low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
            low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

            # Upsample
            net_g = tf.image.resize_images(net_g, low_level_features_shape)
            net_g = tf.concat([net_g, low_level_features], axis=3)
            net_g = slim.conv2d(net_g, 256, [3, 3], scope='conv_3x3_1')
            net_g = slim.conv2d(net_g, 256, [3, 3], scope='conv_3x3_2')

            logit_g = slim.conv2d(net_g, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', activation_fn=None,
                                  normalizer_fn=None)
            logit_g_resize = tf.image.resize_images(logit_g, [img_shape[1] // 2, img_shape[2] // 2])

    # # Fusion
    # net_g = tf.image.resize_images(net_g, [net_l.get_shape().as_list()[1], net_l.get_shape().as_list()[2]])
    # net = tf.concat([net_g, net_l], axis=3)
    # net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_1')
    # net = slim.conv2d(net, 256, [3, 3], scope='conv_3x3_2')
    #
    # logit = slim.conv2d(net, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits', activation_fn=None,
    #                       normalizer_fn=None)
    # logit_resize = tf.image.resize_images(logit, [img_shape[1], img_shape[2]])

    # Fusion logits
    # Binary the uncertainty map according the threshold
    ones = tf.ones(uncertainty.shape)
    zeros = tf.zeros(uncertainty.shape)
    uncer_l = tf.where(uncertainty >= 0.3, ones, zeros)
    uncer_g = 1 - uncer_l
    logit_g_resize_2x = tf.image.resize_images(logit_g_resize, [logit_l_resize.get_shape().as_list()[1],
                                               logit_l_resize.get_shape().as_list()[2]])
    logit_resize = logit_g_resize_2x * uncer_g + logit_l_resize * uncer_l

    label_pred = tf.expand_dims(tf.argmax(logit_l_resize, axis=3, name="prediction"), dim=3)
    logits = [logit_l_resize, logit_g_resize_2x, logit_resize] # [logit2, logit3, logit4, logit5, logit1]

    return label_pred, logits, mask_collects, residual_masks