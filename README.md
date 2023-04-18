# [Automated-Yoga-Pose-Estimation](https://github.com/Sanskarsghyar/Automated-Yoga-Pose-Estimation)

Undergraduation project | Supervisor: Dr. Tushar Sandhan

Develop an AI-based program for the estimation and identification of yoga poses, which can be used as a self-guidance practice framework for individuals to practice yoga postures without getting help from anyone else

The project utilized deep learning algorithms from Mediapipe to extract 33 key points images as feature vector and classify them into different yoga poses using k-Nearest Neighbour (KNN) and Artificial Neural Networks (ANN) with an accuracy of 99% on training dataset & 96% on testing dataset 

## Clone the repository to your local machine:

    git clone https://github.com/Sanskarsghyar/Automated-Pose-Detection

## A] Steps to run Yoga Pose Detection Window:

Navigate to the project directory:

    cd Automated-Pose-Detection

Install the required dependencies:

    pip install -q -r requirements.txt
    
Run the realtime Python window:

    python realtime_try.py


## B] Steps to run Yoga Pose Detection App:
Navigate to the project directory:

    cd Automated-Pose-Detection\deploy_images

Install the required dependencies using the following command:

    pip install -q -r requirements.txt
    
Run the app:

    python app.py

Click this URL http://127.0.0.1:5000/ to open the app

Upload the image and click on predict to predict the yoga pose


## Contributing
If you would like to contribute to this project, you can open an issue or submit a pull request. All contributions are welcome!
