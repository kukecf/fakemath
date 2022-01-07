import cv2 as cv
import tensorflow as tf
from numpy import array
from meta import INPUT_IMAGE_SIZE


# calculate Jaccard index (IoU) of bounding boxes A and B
def bb_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2] + box_a[0], box_b[2] + box_b[0])
    y_b = min(box_a[3] + box_a[1], box_b[3] + box_b[1])
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def bb_area(bb):
    _, _, w, h = bb
    return w * h


def detect_postprocess_bb(bounding_boxes):
    indices_rem = []
    for i in range(len(bounding_boxes) - 1):
        for j in range(i + 1, len(bounding_boxes)):
            iou = bb_iou(bounding_boxes[i], bounding_boxes[j])
            if iou > 0:
                area_i = bb_area(bounding_boxes[i])
                area_j = bb_area(bounding_boxes[j])
                if area_i > area_j:
                    indices_rem.append(j)
                else:
                    indices_rem.append(i)
    bounding_boxes = [i for j, i in enumerate(bounding_boxes) if j not in indices_rem]
    return bounding_boxes


def detect_characters(image_path, kernel=(5, 5), show_results=False):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    opening = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        area = cv.contourArea(contour)
        if 200 < area < 5000:
            x, y, w, h = cv.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
    bounding_boxes = detect_postprocess_bb(bounding_boxes)
    if show_results:
        for (x, y, w, h) in bounding_boxes:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.imshow('Image', img)
        cv.imshow('Opening', opening)
        cv.waitKey(0)

    return sorted(bounding_boxes, key=lambda bb: bb[0])  # sorted by x value


def preprocess_img(image):
    image = tf.image.resize_with_pad(image, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]).numpy()
    image = image / 255.0  # normalization
    image = 1 - image
    return image


def get_digit_images(image_path, kernel=(5, 5)):
    digit_bbs = detect_characters(image_path, kernel=kernel)
    images = []
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
    for bb in digit_bbs:
        (x, y, w, h) = bb
        char_img = tf.convert_to_tensor(binary[y:y + h, x:x + w])
        char_img = array(tf.reshape(char_img, [char_img.shape[0], char_img.shape[1], 1]))
        images.append(preprocess_img(char_img))
    return array(images)

# print("Bounding boxes (x,y,w,h):", detect_characters('notebooks/data/handwritten_ex/20211231_191020.jpg'))
