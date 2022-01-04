from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
from services.card_detector.settings import Settings
from research.object_detection.utils import visualization_utils as vis_util
from utils.utils import encodeImageIntoBase64


class CardsDetector:
    def __init__(self):
        self.settings = Settings

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.settings.MODEL_WEIGHTS_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the image
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def predict(self, image_path):
        # Load the Tensorflow model into memory.
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        sess = tf.Session(graph=self.detection_graph)
        image = cv2.imread(image_path)
        image_expanded = np.expand_dims(image, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        return self.__post_process(boxes, scores, classes, num, image)

    def __post_process(self, boxes, scores, classes, num, image):
        result = scores.flatten()
        res = []
        for idx in range(0, len(result)):
            if result[idx] > self.settings.class_threshold:
                res.append(idx)

        top_classes = classes.flatten()
        # Selecting class 2 and 3
        # top_classes = top_classes[top_classes > 1]
        res_list = [top_classes[i] for i in res]

        class_final_names = [self.settings.class_names_mapping[x] for x in res_list]
        top_scores = [e for l2 in scores for e in l2 if e > self.settings.score_threshold]
        # final_output = list(zip(class_final_names, top_scores))

        # print(final_output)

        # new_classes = classes.flatten()
        new_scores = scores.flatten()

        new_boxes = boxes.reshape(300, 4)

        # get all boxes from an array
        max_boxes_to_draw = new_boxes.shape[0]
        # this is set as a default but feel free to adjust it to your needs

        # iterate over all objects found
        # boundingBox = {}
        # for i in range(min(max_boxes_to_draw, new_boxes.shape[0])):
        #     if new_scores is None or new_scores[i] > min_score_thresh:
        #         boundingBox[class_final_names[i]] = new_boxes[i]
        #         print("Bounding Boxes of", class_final_names[i], new_boxes[i])

        list_of_output = []
        for (name, score, i) in zip(class_final_names, top_scores, range(min(max_boxes_to_draw, new_boxes.shape[0]))):
            val_dict = {}
            val_dict["className"] = name
            val_dict["confidence"] = str(score)
            if new_scores is None or new_scores[i] > self.settings.min_score_thresh:
                val = list(new_boxes[i])
                val_dict["yMin"] = str(val[0])
                val_dict["xMin"] = str(val[1])
                val_dict["yMax"] = str(val[2])
                val_dict["xMax"] = str(val[3])
                list_of_output.append(val_dict)
        # new_boxes = boxes.reshape(100,4)
        # print(new_boxes)
        # print(type(new_boxes))
        # print(new_boxes.shape)
        # print(boxes.shape)
        # Draw the results of the detection (aka 'visulaize the results')

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.settings.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        output_image_name = "output_image_" + str(datetime.now()).split(':')[-1] + ".jpg"
        output_filename = self.settings.OUTPUT_IMAGE_PATH + output_image_name
        cv2.imwrite(output_filename, image)
        open_coded_base64 = encodeImageIntoBase64(output_filename)
        # json_image = dict(zip(img_dict, image_64_encode_list))
        # print(open_output_image)
        # plt.savefig(PATH + '\\' + arr.split('.')[0] + '_labeled.jpg')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # # All the results have been drawn on image. Now display the image.
        # cv2.imshow('Object detector', image)
        #
        # # Press any key to close the image
        # cv2.waitKey(0)
        #
        # # Clean up
        # cv2.destroyAllWindows()
        list_of_output.append({"image": open_coded_base64.decode('utf-8')})
        return list_of_output
