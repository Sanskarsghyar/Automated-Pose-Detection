# Automated-Pose-Detection App

Info

## Features
- User-friendly interface for 
- 

## Prerequisites
Before running the application, make sure you have the following dependencies installed:
- Numpy
- Python 3.10
- MediaPipe
- OpenCV (cv2)
- TensorFlow
- Keras
- Flask

## Clone the repository to your local machine:
    ```bash
    git clone https://github.com/Sanskarsghyar/Automated-Pose-Detection
    ```
    
## A] Steps to access realtime pose detection:

1. Navigate to the project directory:
    ```bash
    cd Automated-Pose-Detection
    ```

2. Install the required dependencies using the following command:
    ```bash
    pip install -r requirements.txt
    ```
    
3. Run the realtimre Python window:
    ```bash
     python realtime_try.py
    ```
    
## B] Steps to run pose detection app:

1. Navigate to the project directory:
    ```bash
    cd Automated-Pose-Detection\deploy_images
    ```

2. Install the required dependencies using the following command:
    ```bash
    pip install -r requirements.txt
    ```
    
3. Run the realtimre Python window:
    ```bash
     python app.py
    ```
4. Click this URL http://127.0.0.1:8080/ to open the app.

5. Upload the image and click on submit to predict the yoga pose

## Model Training
The pre-trained model (mnist.h5) used in this app was trained on the MNIST dataset using a deep neural network. The training code and details of the model architecture can be found in the [Notebook](./notebook/notebook.ipynb) file in this repository.

## Contributing
If you would like to contribute to this project, you can open an issue or submit a pull request. All contributions are welcome!

## License
This project is open-source and available under the MIT License.

## Acknowledgements
This application was developed using the following libraries and frameworks:

- Streamlit: https://streamlit.io/
- OpenCV: https://opencv.org/
- TensorFlow: https://www.tensorflow.org/
- MNIST dataset: http://yann.lecun.com/exdb/mnist/

Special thanks to the authors and contributors of these libraries and datasets for their valuable work.
