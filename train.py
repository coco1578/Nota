from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from math import ceil

from models.ssd_mobilenet import ssd_300
from misc.keras_ssd_loss import SSDLoss
from misc.ssd_box_encode_decode_utils import SSDBoxEncoder
from misc.ssd_batch_generator import BatchGenerator
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
IMG_HEIGHT = 300
IMG_WIDTH = 300
IMG_CHANNELS = 3
subtract_mean = [123, 117, 104]
swap_channels = True # The color channel order in the original SSD is BGR
n_classes = 5 # ['neutral', 'anger', 'surprise', 'smile', 'sad']
scales_voc = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
scales = scales_voc

aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]

two_boxes_for_ar1 = True
steps =[8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
limit_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True

# 1: Build the Keras model

K.clear_session()


def train(args):
    model = ssd_300(mode = 'training',
                  image_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                  n_classes=n_classes,
                  l2_regularization=0.0005,
                  scales=scales,
                  aspect_ratios_per_layer=aspect_ratios,
                  two_boxes_for_ar1=two_boxes_for_ar1,
                  steps=steps,
                  offsets=offsets,
                  limit_boxes=limit_boxes,
                  variances=variances,
                  coords=coords,
                  normalize_coords=normalize_coords,
                  subtract_mean=subtract_mean,
                  divide_by_stddev=None,
                  swap_channels=swap_channels)

    model.load_weights(args.weight_file, by_name=True, skip_mismatch=True)

    predictor_sizes = [model.get_layer('conv11_mbox_conf').output_shape[1:3],
                    model.get_layer('conv13_mbox_conf').output_shape[1:3],
                     model.get_layer('conv14_2_mbox_conf').output_shape[1:3],
                     model.get_layer('conv15_2_mbox_conf').output_shape[1:3],
                     model.get_layer('conv16_2_mbox_conf').output_shape[1:3],
                     model.get_layer('conv17_2_mbox_conf').output_shape[1:3]]

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
    val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

    # 2: Parse the image and label lists for the training and validation datasets. This can take a while.

    # Set the paths to the datasets here.

    images_dir = args.dir_path + '/img'
    annotations_dir = args.dir_path + '/annotations'
    train_image_set_filename = args.dir_path + '/trainval.txt'
    val_image_set_filename = args.dir_path + '/val.txt'

    classes = ['neutral', 'anger', 'surprise', 'smile', 'sad']

    train_dataset.parse_xml(images_dirs=[images_dir],
                      image_set_filenames=[train_image_set_filename],
                      annotations_dirs=[annotations_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)

    val_dataset.parse_xml(images_dirs=[images_dir],
                        image_set_filenames=[val_image_set_filename],
                        annotations_dirs=[annotations_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False
                        )

    # 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

    ssd_box_encoder = SSDBoxEncoder(img_height=IMG_HEIGHT,
                                  img_width=IMG_WIDTH,
                                  n_classes=n_classes,
                                  predictor_sizes=predictor_sizes,
                                  min_scale=None,
                                  max_scale=None,
                                  scales=scales,
                                  aspect_ratios_global=None,
                                  aspect_ratios_per_layer=aspect_ratios,
                                  two_boxes_for_ar1=two_boxes_for_ar1,
                                  steps=steps,
                                  offsets=offsets,
                                  limit_boxes=limit_boxes,
                                  variances=variances,
                                  pos_iou_threshold=0.5,
                                  neg_iou_threshold=0.2,
                                  coords=coords,
                                  normalize_coords=normalize_coords)

    batch_size = args.batch_size

    train_generator = train_dataset.generate(batch_size=batch_size,
                                           shuffle=True,
                                           train=True,
                                           ssd_box_encoder=ssd_box_encoder,
                                           convert_to_3_channels=True,
                                           equalize=False,
                                           brightness=(0.5, 2, 0.5),
                                           flip=0.5,
                                           translate=False,
                                           scale=False,
                                           max_crop_and_resize=(IMG_HEIGHT, IMG_WIDTH, 1, 3),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_pad_and_resize=(IMG_HEIGHT, IMG_WIDTH, 1, 3, 0.5),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_crop=False,
                                           crop=False,
                                           resize=False,
                                           gray=False,
                                           limit_boxes=True,
                                           # While the anchor boxes are not being clipped, the ground truth boxes should be
                                           include_thresh=0.4)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                           shuffle=True,
                                           train=True,
                                           ssd_box_encoder=ssd_box_encoder,
                                           convert_to_3_channels=True,
                                           equalize=False,
                                           brightness=(0.5, 2, 0.5),
                                           flip=0.5,
                                           translate=False,
                                           scale=False,
                                           max_crop_and_resize=(IMG_HEIGHT, IMG_WIDTH, 1, 3),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_pad_and_resize=(IMG_HEIGHT, IMG_WIDTH, 1, 3, 0.5),
                                           # This one is important because the Pascal VOC images vary in size
                                           random_crop=False,
                                           crop=False,
                                           resize=False,
                                           gray=False,
                                           limit_boxes=True,
                                           # While the anchor boxes are not being clipped, the ground truth boxes should be
                                           include_thresh=0.4)

    # Get the number of samples in the training and validations datasets to compute the epoch legnths below.
    n_train_samples = train_dataset.get_n_samples()
    n_val_samples = val_dataset.get_n_samples()

    def lr_schedule(epoch):
        if epoch <= 300:
            return 0.001
        else:
            return 0.0001

    learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule)
    checkpoint_path = args.checkpoint_path + '/ssd300_epoch-{epoch:02d}.h5'
    checkpoint = ModelCheckpoint(checkpoint_path)
    log_path = args.checkpoint_path + '/logs'

    tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=False)

    callbacks = [checkpoint, tensorboard, learning_rate_scheduler]

    epochs = args.epochs
    initial_epoch = args.initial_epoch

    history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=ceil(n_train_samples)/batch_size,
                                verbose=1,
                                initial_epoch=initial_epoch,
                                epochs=epochs,
                                validation_data=val_generator,
                                validation_steps=ceil(n_val_samples)/batch_size,
                                callbacks=callbacks
                                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--dir_path', type=str, help='dataset directory path')
    parser.add_argument('--weight_file', type=str, help='weight file path')
    parser.add_argument('--epochs', type=int, help='number of epoch', default=80)
    parser.add_argument('--initial_epoch', type=int, help='initial epoch', default=0)
    parser.add_argument('--checkpoint_path', type=str, help='path to save checkpoint (model, log)')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)

    args = parser.parse_args()

    train(args)











