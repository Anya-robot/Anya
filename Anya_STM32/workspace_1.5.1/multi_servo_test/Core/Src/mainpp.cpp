/*
 * mainpp.cpp
 *
 *  Created on: Feb 20, 2021
 *      Author: pranav
 */

//#include <main.h>
//#include <mainpp.h>
//#include <ros.h>
//#include <std_msgs/Empty.h>
//#include <std_msgs/ByteMultiArray.h>
//
//
//
//int map(int, int, int, int, int);
//void servo_write(int, int);
//
//// Node:
//ros::NodeHandle nh;
//
//
//int val_0 = 0;
//
//extern TIM_HandleTypeDef htim1;
////extern UART_HandleTypeDef huart2;
//
//
//
//
//
//// Publisher:
//std_msgs::ByteMultiArray array[];
//
//ros::Publisher chatter("chatter", &str_msg);
//char hello[] = "Hello world!";
//
//// Subscriber:
//void led_cb(const std_msgs::Empty &msg) {
//  HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
//}
//ros::Subscriber<std_msgs::Empty> led_sub("toggle_led", &led_cb);
//
//// Setup node:
//void setup(void) {
//  nh.initNode();
//  nh.advertise(chatter);
//  nh.subscribe(led_sub);
//}
//
//// Loop:
//void loop(void) {
//  // Publish message:
//  str_msg.data = hello;
//  chatter.publish(&str_msg);
//
//  nh.spinOnce();
//  HAL_Delay(500);
//}



#include <main.h>
#include <mainpp.h>
#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int32MultiArray.h>

int map(int, int, int, int, int);
void servo_write(int, int);

int val_0 = 0;
int val_1 = 0;
int val_2 = 0;
int val_3 = 0;



extern TIM_HandleTypeDef htim1;


ros::NodeHandle nh;



void servos_go(const std_msgs::Int32MultiArray &msg){



	val_0 = int(msg.data[0]);
	val_1 = int(msg.data[1]);
	val_2 = int(msg.data[2]);
	val_3 = int(msg.data[3]);



	servo_write(val_0, 1);
	servo_write(val_1, 2);
	servo_write(val_2, 3);
	servo_write(val_3, 4);


}

ros::Subscriber<std_msgs::Int32MultiArray> servo_sub("stm32/anya_arm", &servos_go);



// Setup node:
void setup(void) {



  nh.initNode();
  nh.subscribe(servo_sub);


  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4);



}

// Loop:
void loop(void) {
	nh.spinOnce();
//	HAL_Delay(500);
}


int map(int st1, int fn1, int st2, int fn2, int value)
{
    return (1.0*(value-st1))/((fn1-st1)*1.0) * (fn2-st2)+st2;
}

void servo_write(int angle, int channel)
{
	if (channel == 1)
	{
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, map(0,180,1100,5000,angle));
	}
	else if (channel == 2)
	{
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_2, map(0,180,1100,5000,angle));
	}
	else if (channel == 3)
	{
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_3, map(0,180,1100,5000,angle));
	}
	else if (channel == 4)
	{
		__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_4, map(0,180,1100,5000,angle));
	}


}
