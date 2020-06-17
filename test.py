from models.ssd_mobilenet import ssd_300
import cv2
import numpy as np
import os
import argparse
import time
from misc.ssd_box_encode_decode_utils import decode_y
from misc.ssd_batch_generator import BatchGenerator

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def draw_box_and_label(filename, image, boxes, labels, classes):

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
            continue

        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, classes[label-1], (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imwrite(filename, image)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
IMG_HEIGHT = 300
IMG_WIDTH = 300
IMG_CHANNELS = 3
subtract_mean = [127.5,127.5,127.5]
swap_channels = False  # The color channel order in the original SSD is BGR
n_classes = 5 # ['neutral', 'anger', 'surprise', 'smile', 'sad']
scales = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1]
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]]

two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300]
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
limit_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
conf_threshold = 0.5


model = ssd_300("training",
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
                divide_by_stddev=127.5,
                swap_channels=swap_channels)

for layer in model.layers:
    layer.name = layer.name + "_v1"


def test(args):
    model.load_weights(args.weight_file)
    dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

    test_images_dir = args.dir_path + '/img'
    test_image_set_filename = args.dir_path + '/test.txt'

    # The XML parser needs to now what object class names to look for and in which order to map them to integers.
    classes = ['neutral', 'anger', 'surprise', 'smile', 'sad']

    filenames, labels, image_ids = dataset.parse_xml(images_dirs=[test_images_dir],
                                                     image_set_filenames=[test_image_set_filename],
                                                     annotations_dirs=None,
                                                     classes=classes,
                                                     include_classes='all',
                                                     exclude_truncated=False,
                                                     exclude_difficult=False,
                                                     ret=True)

    size = len(filenames)

    for i in range(size):

        image_path = filenames[i]
        ima = cv2.imread(image_path)
        orig_images = []

        orig_images.append(ima)

        image1 = cv2.resize(ima, (IMG_HEIGHT, IMG_WIDTH))
        image1 = image1[np.newaxis, :, :, :]

        input_images = np.array(image1)
        start_time = time.time()
        y_pred = model.predict(input_images)
        print("Time Taken by ssd", time.time() - start_time)

        y_pred_decoded = decode_y(y_pred,
                                  confidence_thresh=0.01,
                                  iou_threshold=0.45,
                                  top_k=18,
                                  input_coords='centroids',
                                  normalize_coords=True,
                                  img_height=IMG_HEIGHT,
                                  img_width=IMG_WIDTH)

        pred_boxes = []
        pred_labels = []
        scores = []

        for box in y_pred_decoded[0]:
            xmin = int(box[-4] * orig_images[0].shape[1] / IMG_WIDTH)
            ymin = int(box[-3] * orig_images[0].shape[0] / IMG_HEIGHT)
            xmax = int(box[-2] * orig_images[0].shape[1] / IMG_WIDTH)
            ymax = int(box[-1] * orig_images[0].shape[0] / IMG_HEIGHT)
            class_id = int(box[0])
            score = box[1]
            pred_boxes.append([xmin, ymin, xmax, ymax])
            pred_labels.append(class_id)
            scores.append(score)

        pred_boxes = np.array(pred_boxes)
        pred_labels = np.array(pred_labels)
        top4_idx = np.argsort(scores)[::-1][:4]

        pred_boxes = pred_boxes[top4_idx]
        pred_labels = pred_labels[top4_idx]

        draw_box_and_label(image_path, ima, pred_boxes, pred_labels, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--dir_path', type=str, help='dataset directory path')
    parser.add_argument('--weight_file', type=str, help='weight file path')

    args = parser.parse_args()
    test(args)