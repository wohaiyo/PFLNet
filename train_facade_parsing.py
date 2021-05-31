from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import glob
import config as cfg
import time
from utils.utils import get_variables_in_checkpoint_file, get_variables_to_restore

from networks.progressive_semantic_completion import  inference_resnet50_progressive_facade_parsing_uncertainty

from image_reader_occlusion import ImageReader
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

def main(argv=None):

    input_size = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    print('Start Train: ' + cfg.TRAIN_DATA_LIST)
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            cfg.TRAIN_DATA_DIR,
            cfg.TRAIN_DATA_LIST,
            input_size,
            cfg.RANDOM_SCALE,
            cfg.RANDOM_MIRROR,
            cfg.RANDOM_CROP_PAD,
            cfg.IGNORE_LABEL,
            cfg.IMG_MEAN,
            coord)
        image_batch, label_batch, mask_batch = reader.dequeue(cfg.BATCH_SIZE)

    # uncertainty threshold
    mask_batch = tf.cast(1 - mask_batch, tf.float32)

    # Define model
    pred_annotation, logits, masks, res_masks, att_maps, feats_col = \
        inference_resnet50_progressive_facade_parsing_uncertainty(tf.concat([image_batch, mask_batch], axis=3))

    # Binary the uncertainty map according the threshold
    ones = tf.ones(mask_batch.shape)
    zeros = tf.zeros(mask_batch.shape)

    # Progressive
    res_masks = []
    for i in range(len(masks)):
        tmp = masks[i]
        tmp = tf.image.resize_images(tmp, mask_batch.get_shape().as_list()[1:3])
        tmp_copy = tf.where(tmp >= 0.8, ones, zeros)        # modified
        masks[i] = tmp_copy

        if i == 0:  res_masks.append(masks[0])
        else:   res_masks.append(masks[i] - masks[i - 1])

    masks.append(masks[-1])
    masks.append(masks[-1])
    masks.append(masks[-1])
    res_masks.append(res_masks[-1])
    res_masks.append(res_masks[-1])
    res_masks.append(res_masks[-1])

    def ce_loss_un_weight(label_batch, pred, uncer):
        gt = tf.squeeze(label_batch, squeeze_dims=[3])
        gt = tf.cast(tf.one_hot(indices=gt, depth=cfg.DATASET_NUM_CLASSESS, on_value=1, off_value=0), dtype=tf.float32)

        pred = tf.nn.softmax(pred)

        if cfg.DATASET_NUM_CLASSESS == 8:
            weights = [0., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        elif cfg.DATASET_NUM_CLASSESS == 12:
            weights = [0., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        elif cfg.DATASET_NUM_CLASSESS == 5:
            weights = [0., 1.0, 1.0, 1.0, 1.0]
        else:
            weights = [0., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        cur_loss = -weights[0] * uncer * gt[:, :, :, 0:1] * tf.log(tf.clip_by_value(pred[:, :, :, 0:1], 1e-10, 1.0))
        for i in range(1, cfg.DATASET_NUM_CLASSESS):
            cur_loss += -weights[i] * uncer * gt[:, :, :, i:i+1] * tf.log(tf.clip_by_value(pred[:, :, :, i:i+1], 1e-10, 1.0))

        return tf.reduce_mean(cur_loss)

    ce_loss = tf.constant(0, tf.float32)
    for i in range(len(logits)):
        # ce_loss += weighted_cross_entropy_loss_artdeco(logits[i], label_batch)
        part_gt = tf.cast(tf.cast(label_batch, tf.float32) * masks[i], tf.int32)
        ce_loss += ce_loss_un_weight(part_gt, logits[i], ones)

    # L2-loss
    l2_loss = [cfg.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables()
                 if 'weights' or 'w' in v.name or 'W' in v.name]
    l2_losses = tf.add_n(l2_loss)

    # Total loss
    loss = ce_loss + l2_losses

    # Summary
    tf.summary.scalar("ce_loss", ce_loss)
    tf.summary.scalar("l2_losses", l2_losses)
    tf.summary.scalar("total_loss", loss)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())

    # Using Poly learning rate policy
    base_lr = tf.constant(cfg.LEARNING_RATE)
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / cfg.NUM_STEPS), cfg.POWER))

    # Optimizer
    opt = tf.train.AdamOptimizer(learning_rate)

    ## Retrieve all trainable variables you defined in your graph
    if cfg.FREEZE_BN:
        tvs = [v for v in tf.trainable_variables()
               if 'beta' not in v.name and 'gamma' not in v.name]
    else:
        tvs = [v for v in tf.trainable_variables()]

    ## Creation of a list of variables with the same shape as the trainable ones
    # initialized with 0s
    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]

    ## Calls the compute_gradients function of the optimizer to obtain... the list of gradients
    gvs = opt.compute_gradients(loss, tvs)

    ## Adds to each element from the list you initialized earlier with zeros its gradient (works because accum_vars and gvs are in the same order)
    accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]

    ## Define the training step (part with variable value update)
    train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    # Set gpu usage
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # Build session
    sess = tf.Session(config=config)
    print("Setting up Saver...")
    # Max number of model
    saver = tf.train.Saver(max_to_keep=cfg.MAX_SNAPSHOT_NUM)
    train_writer = tf.summary.FileWriter(cfg.LOG_DIR + 'train', sess.graph)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Training examples
    _pred = pred_annotation[0]
    _img = image_batch[0]
    _gt = label_batch[0]


    _mask0 = tf.image.resize_nearest_neighbor(masks[0], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]
    _mask1 = tf.image.resize_nearest_neighbor(masks[1], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]
    _mask2 = tf.image.resize_nearest_neighbor(masks[2], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]
    _mask3 = tf.image.resize_nearest_neighbor(masks[3], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]
    _mask4 = tf.image.resize_nearest_neighbor(masks[4], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]

    _mask0_res = tf.image.resize_nearest_neighbor(res_masks[0], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]
    _mask1_res = tf.image.resize_nearest_neighbor(res_masks[1], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]
    _mask2_res = tf.image.resize_nearest_neighbor(res_masks[2], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]
    _mask3_res = tf.image.resize_nearest_neighbor(res_masks[3], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]
    _mask4_res = tf.image.resize_nearest_neighbor(res_masks[4], [cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH])[0]

    # Create save path
    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)
    if not os.path.exists(cfg.SAVE_DIR):
        os.makedirs(cfg.SAVE_DIR)
    if not os.path.exists(cfg.SAVE_DIR + 'temp_img'):
        os.mkdir(cfg.SAVE_DIR + 'temp_img')


    count = 0
    files = os.path.join(cfg.SAVE_DIR + 'model.ckpt-*.index')
    sfile = glob.glob(files)
    if len(sfile) > 0:
        sess.run(tf.global_variables_initializer())
        sfile = glob.glob(files)
        steps = []
        for s in sfile:
            part = s.split('.')
            step = int(part[1].split('-')[1])
            steps.append(step)
        count = max(steps)
        model = cfg.SAVE_DIR + 'model.ckpt-' + str(count)
        print('\nRestoring weights from: ' + model)
        saver.restore(sess, model)
        print('End Restore')
    else:
        # Restore from pre-train on imagenet
        variables = tf.global_variables()
        sess.run(tf.variables_initializer(variables, name='init'))

        if os.path.exists(cfg.PRE_TRAINED_MODEL) or os.path.exists(cfg.PRE_TRAINED_MODEL + '.index'):
            var_keep_dic = get_variables_in_checkpoint_file(cfg.PRE_TRAINED_MODEL)
            # Get the variables to restore, ignoring the variables to fix
            variables_to_restore = get_variables_to_restore(variables, var_keep_dic)
            if len(variables_to_restore) > 0:
                restorer = tf.train.Saver(variables_to_restore)
                restorer.restore(sess, cfg.PRE_TRAINED_MODEL)
                print('Model pre-train loaded from ' + cfg.PRE_TRAINED_MODEL)
            else:
                print('Model inited random.')
        else:
            print('Model inited random.')

        # # Convert RGB -> BGR of resnet_v1_50
        # conv1_rgb = tf.get_variable("conv1_rgb", [7, 7, 3, 64], trainable=False)
        # restorer_fc = tf.train.Saver({'resnet_v1_50/conv1/weights': conv1_rgb})
        # restorer_fc.restore(sess, cfg.PRE_TRAINED_MODEL)
        # sess.run(tf.assign(variables[0], tf.reverse(conv1_rgb, [2])))
        # print('ResNet Conv 1 RGB->BGR')

    print('Start train ...')
    print('---------------Hyper Paras---------------')
    print('-- batch_size: ', cfg.BATCH_SIZE)
    print('-- gradient accumulation: ', cfg.GRADIENT_ACCUMULATION)
    print('-- image height: ', cfg.IMAGE_HEIGHT)
    print('-- image width: ', cfg.IMAGE_WIDTH)
    print('-- learning rate: ', cfg.LEARNING_RATE)
    print('-- GPU: ', cfg.GPU)
    print('-- class num: ', cfg.DATASET_NUM_CLASSESS)
    print('-- total iter: ', cfg.NUM_STEPS)
    print('-- start save step: ', cfg.START_SAVE_STEP)
    print('-- save step every: ', cfg.SAVE_STEP_EVERY)
    print('-- model save num: ', cfg.MAX_SNAPSHOT_NUM)
    print('-- summary interval: ', cfg.SUMMARY_INTERVAL)
    print('-- weight decay: ', cfg.WEIGHT_DECAY)
    print('-- freeze BN: ', cfg.FREEZE_BN)
    print('-- decay rate: ', cfg.POWER)
    print('-- minScale: ', cfg.MIN_SCALE)
    print('-- maxScale: ', cfg.MAX_SCALE)
    print('-- random scale: ', cfg.RANDOM_SCALE)
    print('-- random mirror: ', cfg.RANDOM_MIRROR)
    print('-- random crop: ', cfg.RANDOM_CROP_PAD)
    print('-- pre-trained: ' + cfg.PRE_TRAINED_MODEL)
    print('----------------End---------------------')
    fcfg = open(cfg.SAVE_DIR + 'cfg.txt', 'w')
    fcfg.write('-- batch_size: ' + str(cfg.BATCH_SIZE) + '\n')
    fcfg.write('-- gradient accumulation: ' + str(cfg.GRADIENT_ACCUMULATION) + '\n')
    fcfg.write('-- image height: ' + str(cfg.IMAGE_HEIGHT) + '\n')
    fcfg.write('-- image width: ' + str(cfg.IMAGE_WIDTH) + '\n')
    fcfg.write('-- learning rate: ' + str(cfg.LEARNING_RATE) + '\n')
    fcfg.write('-- GPU: ' + str(cfg.GPU) + '\n')
    fcfg.write('-- class num: ' + str(cfg.DATASET_NUM_CLASSESS) + '\n')
    fcfg.write('-- total iter: ' + str(cfg.NUM_STEPS) + '\n')
    fcfg.write('-- start save step: ' + str(cfg.START_SAVE_STEP) + '\n')
    fcfg.write('-- save step every: ' + str(cfg.SAVE_STEP_EVERY) + '\n')
    fcfg.write('-- model save num: ' + str(cfg.MAX_SNAPSHOT_NUM) + '\n')
    fcfg.write('-- summary interval: ' + str(cfg.SUMMARY_INTERVAL) + '\n')
    fcfg.write('-- weight decay: ' + str(cfg.WEIGHT_DECAY) + '\n')
    fcfg.write('-- freeze BN: ' + str(cfg.FREEZE_BN) + '\n')
    fcfg.write('-- decay rate: ' + str(cfg.POWER) + '\n')
    fcfg.write('-- minScale: ' + str(cfg.MIN_SCALE) + '\n')
    fcfg.write('-- maxScale: ' + str(cfg.MAX_SCALE) + '\n')
    fcfg.write('-- random scale: ' + str(cfg.RANDOM_SCALE) + '\n')
    fcfg.write('-- random mirror: ' + str(cfg.RANDOM_MIRROR) + '\n')
    fcfg.write('-- random crop: ' + str(cfg.RANDOM_CROP_PAD) + '\n')
    fcfg.write('-- pre-trained: ' + str(cfg.PRE_TRAINED_MODEL) + '\n')
    fcfg.close()

    last_summary_time = time.time()

    # iteration number of each epoch
    record = cfg.DATASET_SIZE / cfg.BATCH_SIZE

    running_count = count
    epo = int(count / record)

    train_start_time = time.time()

    # Change the graph for read only
    sess.graph.finalize()

    # Start training
    while running_count < cfg.NUM_STEPS:
        time_start = time.time()
        itr = 0
        while itr < int(record):
            itr += 1
            running_count += 1

            # More than total iter, stopping training
            if running_count > cfg.NUM_STEPS:
                break

            feed_dict = {step_ph: (running_count)}

            # Save summary and example images
            now = time.time()
            if now - last_summary_time > cfg.SUMMARY_INTERVAL:
                summary_str = sess.run(summary_op, feed_dict={step_ph: running_count})
                train_writer.add_summary(summary_str, running_count)
                last_summary_time = now

                # Save tmp results
                score_map, img, gt, \
                    mask0, mask1, mask2, mask3, mask4,\
                    mask_res0, mask_res1, mask_res2, mask_res3, mask_res4, \
                    = sess.run([_pred, _img, _gt,
                                               _mask0, _mask1, _mask2, _mask3, _mask4,
                                               _mask0_res, _mask1_res, _mask2_res, _mask3_res, _mask4_res], feed_dict=feed_dict)

                # score_map, img, gt = sess.run([_pred, _img, _gt], feed_dict=feed_dict)e
                img = np.array(img + cfg.IMG_MEAN, np.uint8)
                score_map = score_map * 20
                gt = gt * 20

                save_temp = np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 3, 3), np.uint8)
                save_temp[0:cfg.IMAGE_HEIGHT, 0:cfg.IMAGE_WIDTH, :] = img
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH:cfg.IMAGE_WIDTH * 2, :] = gt
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 2:cfg.IMAGE_WIDTH * 3, :] = score_map
                cv2.imwrite(cfg.SAVE_DIR + 'temp_img/' + str(now) + 'result.jpg', save_temp)

                mask0 = mask0 * 255
                mask1 = mask1 * 255
                mask2 = mask2 * 255
                mask3 = mask3 * 255
                mask4 = mask4 * 255
                save_temp = np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 5, 3), np.uint8)
                save_temp[0:cfg.IMAGE_HEIGHT, 0:cfg.IMAGE_WIDTH, :] = mask0
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH:cfg.IMAGE_WIDTH * 2, :] = mask1
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 2:cfg.IMAGE_WIDTH * 3, :] = mask2
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 3:cfg.IMAGE_WIDTH * 4, :] = mask3
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 4:cfg.IMAGE_WIDTH * 5, :] = mask4
                cv2.imwrite(cfg.SAVE_DIR + 'temp_img/' + str(now) + '_mask1.jpg', save_temp)

                mask0 = mask_res0 * 255
                mask1 = mask_res1 * 255
                mask2 = mask_res2 * 255
                mask3 = mask_res3 * 255
                mask4 = mask_res4 * 255
                save_temp = np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 5, 3), np.uint8)
                save_temp[0:cfg.IMAGE_HEIGHT, 0:cfg.IMAGE_WIDTH, :] = mask0
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH:cfg.IMAGE_WIDTH * 2, :] = mask1
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 2:cfg.IMAGE_WIDTH * 3, :] = mask2
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 3:cfg.IMAGE_WIDTH * 4, :] = mask3
                save_temp[0:cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH * 4:cfg.IMAGE_WIDTH * 5, :] = mask4
                cv2.imwrite(cfg.SAVE_DIR + 'temp_img/' + str(now) + '_mask_res.jpg', save_temp)

            time_s = time.time()

            # Run the zero_ops to initialize it
            sess.run(zero_ops)

            # Accumulate the gradients 'n_minibatches' times in accum_vars using accum_ops
            for i in range(cfg.GRADIENT_ACCUMULATION):
                sess.run(accum_ops, feed_dict=feed_dict)
            train_loss, ls_ce, ls_l2, lr = sess.run([loss, ce_loss, l2_losses, learning_rate], feed_dict=feed_dict)

            # Run the train_step ops to update the weights based on your accumulated gradients
            sess.run(train_step, feed_dict=feed_dict)

            time_e = time.time()

            print("Epo: %d, Step: %d, Train_loss:%g, ce: %g, l2:%g,  lr:%g, time:%g" %
                  (epo, running_count, train_loss, ls_ce, ls_l2, lr, time_e - time_s))

            # Save step model
            if (running_count % cfg.SAVE_STEP_EVERY) == 0 \
                    and running_count >= cfg.START_SAVE_STEP:
                saver.save(sess, cfg.SAVE_DIR + 'model.ckpt', int(running_count))
                print('Model has been saved:' + str(running_count))
                files = os.path.join(cfg.SAVE_DIR + 'model.ckpt-*.data-00000-of-00001')
                sfile = glob.glob(files)
                if len(sfile) > cfg.MAX_SNAPSHOT_NUM:
                    steps = []
                    for s in sfile:
                        part = s.split('.')
                        re = int(part[1].split('-')[1])
                        steps.append(re)
                    re = min(steps)
                    model = cfg.SAVE_DIR + 'model.ckpt-' + str(re)
                    os.remove(model + '.data-00000-of-00001')
                    os.remove(model + '.index')
                    os.remove(model + '.meta')
                    print('Remove Model:' + model)

        epo += 1
        time_end = time.time()
        print('Epo ' + str(epo) + ' use time: ' + str(time_end - time_start))

    # Finish training
    train_end_time = time.time()
    print('Train total use: ' + str((train_end_time-train_start_time) / 3600) + ' h')
    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    tf.app.run()

