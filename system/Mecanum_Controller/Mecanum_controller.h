/**
* This file is part of Mecanum-follower.
* Copyright (C) 2021 Gabriel Fischer Abati - Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: João Carlos Virgolino Soares, Marcelo Gattass and Marco Antonio Meggiolaro
* For more information see https://github.com/Master-Fischer/Mecanum_Follower.
* Please report suggestions and comments to fischerabati@gmail.com
**/

#ifndef MECANUM_CONTROLLER
#define MECANUM_CONTROLLER 2

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <iostream>

#include <vector>
#include <fstream>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"


class Mecanum_controller
{
    public:
        Mecanum_controller(); //input setpoint
        ~Mecanum_controller();
        std::string calculate(cv::Point s,cv::Point p,float depth);
        
    private:
        cv::Point setpoint;
        float Kp_im=0.0;
        float Kp_depth=0.18;
        float depth_th=800; //mm 
        double dist(cv::Point p1,cv::Point p2);
};

#endif