/**
* This file is part of Mecanum-follower.
* Copyright (C) 2021 Gabriel Fischer Abati - Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: João Carlos Virgolino Soares, Marcelo Gattass and Marco Antonio Meggiolaro
* For more information see https://github.com/Master-Fischer/Mecanum_Follower.
* Please report suggestions and comments to fischerabati@gmail.com
**/


#include "Mecanum_controller.h"

Mecanum_controller::Mecanum_controller(){}
//=========================================
Mecanum_controller::~Mecanum_controller(){};
//=========================================
double Mecanum_controller::dist(cv::Point p1,cv::Point p2){return (p1.y-p2.y)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y);}
//==========================================
std::string Mecanum_controller::calculate(cv::Point s,cv::Point p,float depth)
{   
    std::string msg="";
    //calculate the distance error from the centroid p from the setpoint s
    float error = dist(s,p);
    float error_depth=(depth-this->depth_th);
    if (error_depth==INFINITY || error_depth<0)
    {
        error_depth=0;
        return "S 0";
    }
    std::cout<<"error_depth: "<<error_depth;

    //get the angle from the centroid from the middle secction of the frame
    //the angle is then associate with 3 arbitrary region in order to tell which direction 
    // the robot must go
    float theta=atan2(p.y-s.y,p.x-s.x);
    theta=theta*(180/M_PI);  //convert rad to degree 
    theta*=-1; // Y axis is inverted in computer windows, Y goes down, so invert the angle.

    //Angle returned as:
    //                      90
    //            135                45
    //
    //       180          Origin           0
    //
    //           -135                -45
    //
    //                     -90

    //if(theta<0) { theta=abs(theta)+180;}
    std::cout<<"  theta: "<<theta;

    //calculate output result of a P controller
    int out=this->Kp_im*error + this->Kp_depth*error_depth;

    //creating message to the low level controller
    
    if(theta>-45 && theta<=45)
    {
        //go NE
        std::cout<<"  go NE "<<" out: "<<out<<std::endl;
        msg="I "+ std::to_string(out);
    }
    else if (theta>45 && theta<135 || theta>-135 && theta<=-45)
    {
        //go foward 
        std::cout<<"  go N" <<" out: "<<std::endl;
        msg="N "+ std::to_string(out);
    }
    else if (theta>=135 && theta<180 || theta<-135 && theta>-179)
    {
        //go NW
        std::cout<<" go NW" <<" out: "<<std::endl;
        msg="G "+ std::to_string(out);
    }
    else
    {
        std::cout<<"STOP!"<<std::endl;
        msg="S 0";
    }

    return msg;
}