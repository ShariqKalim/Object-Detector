# import cv2 as cv
# import numpy as np

# # Distance constants
# KNOWN_DISTANCE = 45  # INCHES
# PERSON_WIDTH = 16  # INCHES
# MOBILE_WIDTH = 3.0  # INCHES

# # Object detector constants
# CONFIDENCE_THRESHOLD = 0.4
# NMS_THRESHOLD = 0.3

# # Colors for object detected
# COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255),
#           (255, 255, 0), (0, 255, 0), (255, 0, 0)]
# GREEN = (0, 255, 0)
# BLACK = (0, 0, 0)

# # Defining fonts
# FONTS = cv.FONT_HERSHEY_COMPLEX

# # Getting class names from classes.txt file
# class_names = []
# with open("classes.txt", "r") as f:
#     class_names = [cname.strip() for cname in f.readlines()]

# # Setting up OpenCV YOLO net
# yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# model = cv.dnn_DetectionModel(yoloNet)
# model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# # Object detector function/method


# def object_detector(image):
#     classes, scores, boxes = model.detect(
#         image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
#     # Creating an empty list to add object data
#     data_list = []
#     for classid, score, box in zip(classes, scores, boxes):
#         # classid can be an array containing multiple class IDs, so we need to handle it accordingly
#         if isinstance(classid, np.ndarray):
#             class_id = classid[0]
#         else:
#             class_id = classid

#         # Define color of each object based on its class ID
#         color = COLORS[int(class_id) % len(COLORS)]

#         label = "%s : %f" % (class_names[class_id], score)

#         # Draw rectangle and label on object
#         cv.rectangle(image, box, color, 2)
#         cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

#         # Getting the data
#         # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
#         if class_id == 0:  # person class id
#             data_list.append(
#                 [class_names[class_id], box[2], (box[0], box[1] - 2)])
#         elif class_id == 67:  # cell phone class id
#             data_list.append(
#                 [class_names[class_id], box[2], (box[0], box[1] - 2)])
#         # If you want to include more classes, you have to simply add more [elif] statements here
#         # Returning list containing the object data.
#     return data_list


# def focal_length_finder(measured_distance, real_width, width_in_rf):
#     focal_length = (width_in_rf * measured_distance) / real_width
#     return focal_length


# def distance_finder(focal_length, real_object_width, width_in_frame):
#     distance = (real_object_width * focal_length) / width_in_frame
#     return distance


# # Reading the reference image from dir
# ref_person = cv.imread('ReferenceImages/image14.png')
# ref_mobile = cv.imread('ReferenceImages/image4.png')

# mobile_data = object_detector(ref_mobile)
# mobile_width_in_rf = mobile_data[1][1]

# person_data = object_detector(ref_person)
# person_width_in_rf = person_data[0][1]

# print(
#     f"Person width in pixels: {person_width_in_rf}, Mobile width in pixels: {mobile_width_in_rf}")

# # Finding focal length
# focal_person = focal_length_finder(
#     KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

# focal_mobile = focal_length_finder(
#     KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

# cap = cv.VideoCapture(0)
# while True:
#     ret, frame = cap.read()

#     data = object_detector(frame)
#     for d in data:
#         if d[0] == 'person':
#             distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
#             x, y = d[2]
#         elif d[0] == 'cell phone':
#             distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
#             x, y = d[2]
#         cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
#         cv.putText(frame, f'Dis: {round(distance, 2)} inch',
#                    (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

#     cv.imshow('frame', frame)

#     key = cv.waitKey(1)
#     if key == ord('q'):
#         break

# cv.destroyAllWindows()
# cap.release()


import cv2 as cv
import numpy as np

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constants
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255),
          (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# Getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Setting up OpenCV YOLO net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# Object detector function/method


def object_detector(image):
    classes, scores, boxes = model.detect(
        image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # Creating an empty list to add object data
    data_list = []
    for classid, score, box in zip(classes, scores, boxes):
        # classid can be an array containing multiple class IDs, so we need to handle it accordingly
        if isinstance(classid, np.ndarray):
            class_id = classid[0]
        else:
            class_id = classid

        # Define color of each object based on its class ID
        color = COLORS[int(class_id) % len(COLORS)]

        label = "%s : %f" % (class_names[class_id], score)

        # Draw rectangle and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # Getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if class_id == 0:  # person class id
            data_list.append(
                [class_names[class_id], box[2], (box[0], box[1] - 2)])
        elif class_id == 67:  # cell phone class id
            data_list.append(
                [class_names[class_id], box[2], (box[0], box[1] - 2)])
        # If you want to include more classes, you have to simply add more [elif] statements here
        # Returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length


def distance_finder(focal_length, real_object_width, width_in_frame):
    # Convert the real object width from inches to meters
    real_object_width_meter = real_object_width * 0.0254

    # Calculate the distance in meters
    distance_meter = (real_object_width_meter * focal_length) / width_in_frame
    return distance_meter


# Reading the reference image from dir
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(
    f"Person width in pixels: {person_width_in_rf}, Mobile width in pixels: {mobile_width_in_rf}")

# Finding focal length
focal_person = focal_length_finder(
    KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(
    KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} meters',
                   (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
