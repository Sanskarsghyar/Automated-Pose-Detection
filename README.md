# [Automated-Yoga-Pose-Estimation](https://github.com/Sanskarsghyar/Automated-Yoga-Pose-Estimation)

Undergraduation project | Supervisor: Dr. Tushar Sandhan

Develop an AI-based program for the estimation and identification of yoga poses, which can be used as a self-guidance practice framework for individuals to practice yoga postures without getting help from anyone else

- Extracted the 33 important keypoints as feature vectors from the individual’s skeleton using Mediapipe
- Enhanced features by incorporating 8 essential joint angles and normalized keypoints w.r.t. pose center
- Trained KNN and ANN model, and achieved an accuracy of 99% for training and 96% for testing dataset

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
