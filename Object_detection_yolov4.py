import cv2
import os
import numpy as np
import time
import datetime
import os
import shutil


class ImportYoloGraph():
    def __init__(self, modelpath, configpath, labelpath, Width, Height):
        weights_filename = modelpath
        config_filename = configpath
        print(weights_filename)
        print(config_filename)

        try:
            self.net = cv2.dnn.readNet(weights_filename, config_filename)
        except Exception as e:
            print(e)

    def get_output_layers(self, my_net):
        layer_names = my_net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in my_net.getUnconnectedOutLayers()]
        return output_layers

    def detectGetCoordinates(self, image_data):
        Width = image_data.shape[1]
        Height = image_data.shape[0]
        print(Width, Height)
        scale = 0.00392   # 1 / 255

        blob = cv2.dnn.blobFromImage(image_data, scale, (Width, Height), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.get_output_layers(self.net))
        confidences = []
        boxes = []
        centers = []
        classes = []

        for out in outs:
            for detection in out:
                # x, y, w, h, pc, c
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2

                # Get all results without 0 confidence
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence != 0:
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    centers.append([center_x, center_y])
                    classes.append(class_id)

        return confidences, boxes, centers, classes


if __name__ == "__main__":

    YoloWeightPath = 'model\\YOLO\\yolov4_best.weights'
    YoloCfgPath = 'model\\YOLO\\yolov4.cfg'
    YoloNamePath = 'model\\YOLO\\insect.names'
    Yolo = ImportYoloGraph(YoloWeightPath, YoloCfgPath, YoloNamePath, 3200, 2400)

    #
    # object extraction and clean
    #
    Input_path = 'dataset/2-loc-all'

    Obj_path = 'Result/2-loc-all_YOLOv4_v2'
    os.makedirs(Obj_path, exist_ok=True)
    shutil.copy(os.path.basename(__file__), Obj_path + "/program.py")

    # Month_list = os.listdir(Input_path)
    Month_list = ['2021_01', '2021_02', '2021_03', '2021_04', '2021_05', '2021_06', '2021_07', '2021_08', '2021_09',
                  '2021_10', '2021_11', '2021_12',
                  '2022_01', '2022_02', '2022_03', '2022_04', '2022_05', '2022_06', '2022_07', '2022_08', '2022_09',
                  '2022_10', '2022_11', '2022_12',
                  ]
    for Month in Month_list:
        print(Month)

        os.makedirs(Obj_path + "/" + Month, exist_ok=True)
        os.makedirs(Obj_path + "/" + Month + "/unknown", exist_ok=True)
        os.makedirs(Obj_path + "/" + Month + "/whitefly", exist_ok=True)
        for ins in Stage3List:
            os.makedirs(Obj_path + "/" + Month + "/" + ins, exist_ok=True)

        image_path = os.listdir(Input_path + "/" + Month)
        for i, image in enumerate(image_path):

            print("\r" + "{} / {}  {} ".format(i + 1, len(image_path), image), end='')

            location = image.split('_')[0]
            node = image.split('_')[1]
            year = image.split('_')[2]
            moth = image.split('_')[3]
            day = image.split('_')[4].split(" ")[0]
            hour = image.split('_')[4].split(" ")[1]
            minute = image.split('_')[5]

            pre_save_name = location + "_" + node + "_" + year + "_" + moth + "_" + day + "_" + hour + "_" \
                            + minute + "_"

            detect_image = cv2.resize(cv2.imread(Input_path + "/" + Month + "/" + image), (3200, 2400))
            crop_image = detect_image.copy()
            no_nms_scores, no_nms_boxes, centers, class_names = Yolo.detectGetCoordinates(detect_image)

            indices = cv2.dnn.NMSBoxes(no_nms_boxes, no_nms_scores, 0.05, 0.05)

            obj_list = []
            for a in indices:
                a = a[0]
                box = no_nms_boxes[a]
                x, y, w, h = box
                side = np.maximum(w, h)

                if side <= 128:
                    side = side + 5

                x1 = x + (w - side) / 2
                y1 = y + (h - side) / 2
                x2 = x1 + side
                y2 = y1 + side

                c1 = int(max(0, y1))
                c2 = int(y2)
                c3 = int(max(0, x1))
                c4 = int(x2)

                c1 = max(0, c1)
                c2 = min(2400, c2)
                c3 = max(0, c3)
                c4 = min(3200, c4)

                cv2.rectangle(detect_image, (c3, c1), (c4, c2), (255, 0, 0), 2)
                obj_list.append(crop_image[c1:c2, c3:c4])
            cv2.imwrite(Obj_path + "/" + Month + "/" + image, detect_image)

            save_index = 0
            for obj in obj_list:
                # obj = 你框到的東西
                pass

        print("")
