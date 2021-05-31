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


    # For ablation study of different steps
    # mask_collects = [mask0, mask8]
    # mask_collects = [mask0, mask4, mask8]
    mask_collects = [mask0, mask3, mask5, mask8]  # 0, 3, 5, 8
    # mask_collects = [mask0, mask3, mask4, mask5, mask8]

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

    # progressive masks for learning
    residual_masks = [mask0]
    for i in range(1, len(mask_collects)):
        residual_masks.append((mask_collects[i] - mask_collects[i - 1]))

    return mask_collects, residual_masks

# Progressive parsing with uncertainty map

def inference_resnet50_progressive_facade_parsing_uncertainty(image_batch, is_training=True):

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
