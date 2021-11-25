# Mecanum Follower

The objective of this project is to make a mecanum robot follow a person using deep learning
techniques. The proposed system uses a yoloV3 object detection model with SORT algorithm for tracking, by Cong Ma. A Kinect camera is used in order for the  robot to estimate the distance between itself and the human. The depth distance and distance between the centroid and the image center point is then measured and applied to a  simple control law to drive the robot motors.


<img src="images/system/robot.jpeg"
width="773" height="489" /></a>
<!--img src="images/MOT20-01_CYTi.jpg" 
width="960" height="540" /></a!-->

# License

Mecanum_Follower is released under a GPLv3 License.

If you use Mecanum_Follower in an academic work, please cite:

    @PROCEEDINGS{ABATI2021,
      title={PEOPLE FOLLOWING SYSTEM FOR HOLONOMIC ROBOTS USING AN RGB-D SENSOR},
      author={Abati, G. F.,Soares,J. C. V., Gattass, M., & Meggiolaro, M. A.},
      Conference={26th International Congress of Mechanical Engineering},
      doi = {},
      year={2021}
     }
     
# Building Mecanum_Follower
- Clone the repository:
```
git clone https://github.com/Master-Fischer/Mecanum_Follower
```

Install libfreenect2 for KinectV2 RGB-D sensor:
```
./install_libfreenect2.sh
```

- Build the system:
```
cd Mecanum_Follower/system
mkdir build
cd build
cmake ..
make
```


# Run Mecanum Follower 
```
./yoloDepth
```





















