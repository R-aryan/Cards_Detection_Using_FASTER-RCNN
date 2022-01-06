# Cards Detection Using FASTER RCNN

- End to end object detection project using Faster RCNN.
- The training is done using [TFOD1.14(Tensorflow object detection) framework](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/).
- The application is served as an REST API using Flask.
- Faster RCNN paper can be found [here](https://arxiv.org/pdf/1506.01497.pdf).


# Steps to run the application 

### Setting up  virtual environment.

- What is [**Virtual Environment in python ?**](https://www.geeksforgeeks.org/python-virtual-environment/)
- [Create virtual environment in python](https://www.geeksforgeeks.org/creating-python-virtual-environment-windows-linux/)
- [Create virtual environment Anaconda](https://www.geeksforgeeks.org/set-up-virtual-environment-for-python-using-anaconda/)
- create a virtual environment and install [requirements.txt](https://github.com/R-aryan/Cards_Detection_Using_FASTER-RCNN/blob/develop/requirements.txt)

> pip install -r requirements.txt

- After setting up the virtual environment download the trained weights from [here](https://drive.google.com/file/d/1AckrDU2sNq9l1tW-Pb2xWYnFCcEVgNkx/view?usp=sharing).
- After downloading the trained weights place it under the directory [**services/card_detector/application/ai/weights/exported_inference_graph/**](https://github.com/R-aryan/Cards_Detection_Using_FASTER-RCNN/tree/develop/services/card_detector/application/ai/weights/exported_inference_graph)
- After performing the above steps go to [services/card_detector/api](https://github.com/R-aryan/Cards_Detection_Using_FASTER-RCNN/tree/develop/services/card_detector/api) and run **app.py**
> python app.py