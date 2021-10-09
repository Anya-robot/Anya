/*
 * mainpp.cpp
 *
 *  Created on: Feb 20, 2021
 *      Author: pranav
 */

#include <main.h>
#include <mainpp.h>
#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/ByteMultiArray.h>

int map(int, int, int, int, int);
void servo_write(int);

//extern int map(int, int, int, int, int);
//extern void servo_write(int);

extern TIM_HandleTypeDef htim1;

extern UART_HandleTypeDef huart2;
//DMA_HandleTypeDef hdma_usart2_rx;
//DMA_HandleTypeDef hdma_usart2_tx;
//
//
//
//
//void SystemClock_Config(void);
//static void MX_GPIO_Init(void);
//static void MX_DMA_Init(void);
//static void MX_USART2_UART_Init(void);
//static void MX_TIM1_Init(void);



// Node:
ros::NodeHandle nh;

// Publisher:
//std_msgs::ByteMultiArray array[];


//ros::Publisher chatter("chatter", &str_msg);
//char hello[] = "Hello world!";


void servo_go(const std_msgs::ByteMultiArray &msg){

	int val = 0;
	val = int(msg.data[0]);

//	val = (val << 8) + msg.data[0];

	servo_write(val);

}

ros::Subscriber<std_msgs::ByteMultiArray> servo_sub("chatter", &servo_go);

// Subscriber:
//void led_cb(const std_msgs::Empty &msg) {
//  HAL_GPIO_TogglePin(GPIOA, GPIO_PIN_5);
//}
//ros::Subscriber<std_msgs::Empty> led_sub("toggle_led", &led_cb);

// Setup node:
void setup(void) {


//  HAL_Init();
//
//  SystemClock_Config();
//
//
//  MX_GPIO_Init();
//  MX_DMA_Init();
//  MX_USART2_UART_Init();
//  MX_TIM1_Init();

//  MX_TIM1_Init();
  nh.initNode();
//  nh.advertise(chatter);
//  nh.subscribe(led_sub);

  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
  nh.subscribe(servo_sub);

}

// Loop:
void loop(void) {
  // Publish message:
//  str_msg.data = hello;
//  chatter.publish(&str_msg);
  for(int i = 0; i <= 180; i++)
  {
	servo_write(i);
	HAL_Delay(10);
  }
  for(int i = 180; i >= 0; i--)
  {
    servo_write(i);
    HAL_Delay(10);
  }
  nh.spinOnce();
  HAL_Delay(500);
}


int map(int st1, int fn1, int st2, int fn2, int value)
{
    return (1.0*(value-st1))/((fn1-st1)*1.0) * (fn2-st2)+st2;
}

void servo_write(int angle)
{

	__HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, map(0,180,2000,4000,angle));


}
