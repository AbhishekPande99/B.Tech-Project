## B.Tech-Project ECN-400A

Facial Feature Detection and Reenactment Onto Computer Generated Imagery

##### Instructions to run the code:
```
git clone https://github.com/AbhishekPande99/B.Tech-Project.git
```

Install the pip dependencies and enter the code directory
```
pip3 install -r requirements.txt
cd B.Tech-Project
```

To get facial landmarks of the image.
```
python3 landmarks.py shape_predictor_68_face_landmarks.dat images/{image.jpg} 
```

To get the head pose estimation of the image.
```
python3 head_pose.py shape_predictor_68_face_landmarks.dat images/{image.jpg}
```
