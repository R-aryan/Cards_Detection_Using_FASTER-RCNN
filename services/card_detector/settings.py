import os

from services.card_detector.application.ai.research.object_detection.utils import label_map_util


class Settings:
    sep = None
    if os.name == 'nt':
        sep = "\\"
    else:
        sep = "/"

    PROJ_NAME = 'Cards_Detection_Using_FASTER-RCNN'
    root_path = os.getcwd().split(PROJ_NAME)[0] + PROJ_NAME + sep
    APPLICATION_PATH = root_path + "services" + sep + "card_detector" + sep + "application" + sep
    print(APPLICATION_PATH)
    # setting up logs path
    LOGS_DIRECTORY = root_path + "services" + sep + "card_detector" + sep + "logs" + sep + "logs.txt"

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    MODEL_WEIGHTS_PATH = APPLICATION_PATH + "ai" + sep + "weights" + sep + "exported_inference_graph" + sep + "frozen_inference_graph.pb "

    # Path to label map file
    PATH_TO_LABELS = APPLICATION_PATH + "ai" + sep + "research" + sep + "data" + sep + "labelmap.pbtxt"
    print(PATH_TO_LABELS)

    NUM_CLASSES = 6

    # input image path
    INPUT_IMAGE_PATH = root_path + "services\\card_detector\\images\\input_images\\"
    # output image path
    OUTPUT_IMAGE_PATH = root_path + "services\\card_detector\\images\\output_images\\"

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    class_names_mapping = {
        1: "Nine", 2: "Ten", 3: "jack", 4: "queen", 5: "King", 6: "Ace"
    }

    score_threshold = 0.30
    class_threshold = 0.40
    min_score_thresh = .30
    image_extension = ".jpg"
