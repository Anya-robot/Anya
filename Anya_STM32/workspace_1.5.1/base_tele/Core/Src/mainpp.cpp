#include <main.h>
#include <mainpp.h>
#include <ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/ByteMultiArray.h>

int map(int, int, int, int, int);

void stop();

void start();

void direction(int, int);

void motors_go(int);

void accelerate(int);

void decelerate(int);

int linear = 0;
int angular = 0;

int l_flag = 0;
int r_flag = 0;



int current_speed = 49999;

int arr = 39999;


extern TIM_HandleTypeDef htim10;
extern TIM_HandleTypeDef htim11;



ros::NodeHandle nh;



void wheels_go(const std_msgs::ByteMultiArray &msg){



	linear = int(msg.data[0]);
	angular = int(msg.data[1]);

	if ((angular == 0) and (linear == 0)){
		stop();
		return;
	}

	start();

	if (angular == 0){

		if (linear > 0){

			if (((r_flag == 0) and (l_flag == 0)) or ((r_flag == 1) and (l_flag == 1)))
			{
				direction(1, 1);
				motors_go(linear);
			}

			else
			{
				motors_go(0);
				direction(1, 1);
				motors_go(linear);
				r_flag = 1;
				l_flag = 1;
			}


		}

		else if (linear < 0){

			if (((r_flag == 0) and (l_flag == 0)) or ((r_flag == 2) and (l_flag == 2)))
			{
				direction(0, 0);
				motors_go((-1*linear));
			}

			else
			{
				motors_go(0);
				direction(0, 0);
				motors_go((-1*linear));
				r_flag = 2;
				l_flag = 2;
			}
		}
	}

	if (linear == 0){

		if (angular > 0){

			if (((r_flag == 0) and (l_flag == 0)) or ((r_flag == 1) and (l_flag == 2)))
			{
				direction(0, 1);
				motors_go(angular);
			}

			else
			{
				motors_go(0);
				direction(0, 1);
				motors_go(angular);
				r_flag = 1;
				l_flag = 2;
			}

		}

		else if (angular < 0){

			if (((r_flag == 0) and (l_flag == 0)) or ((r_flag == 1) and (l_flag == 2)))
			{
				direction(1, 0);
				motors_go((-1*angular));
			}

			else
			{
				motors_go(0);
				direction(1, 0);
				motors_go((-1*angular));
				r_flag = 1;
				l_flag = 2;
			}
		}
	}








}

ros::Subscriber<std_msgs::ByteMultiArray> base_sub("stm32/anya_base", &wheels_go);

//ros::Subscriber<std_msgs::Int32MultiArray> servo_sub("chatter", &servos_go);



// Setup node:
void setup(void) {



  nh.initNode();
  nh.subscribe(base_sub);


  HAL_TIM_PWM_Start(&htim10, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start(&htim11, TIM_CHANNEL_1);

}

// Loop:
void loop(void) {
	nh.spinOnce();
	HAL_Delay(500);
}


int map(int st1, int fn1, int st2, int fn2, int value)
{
    return (1.0*(value-st1))/((fn1-st1)*1.0) * (fn2-st2)+st2;
}



void stop()
{
//	decelerate(49999);
	__HAL_TIM_SET_COMPARE(&htim10, TIM_CHANNEL_1, 0);
	__HAL_TIM_SET_COMPARE(&htim11, TIM_CHANNEL_1, 0);

	l_flag = 0;
	r_flag = 0;
}

void start()
{
	__HAL_TIM_SET_COMPARE(&htim10, TIM_CHANNEL_1, 3000);
	__HAL_TIM_SET_COMPARE(&htim11, TIM_CHANNEL_1, 3000);

}

void direction(int dir1, int dir2)
{
	if (dir1 == 0){
		HAL_GPIO_WritePin(DIR_1_GPIO_Port, GPIO_PIN_14, GPIO_PIN_RESET);
	}
	else if (dir1 == 1){
		HAL_GPIO_WritePin(DIR_1_GPIO_Port, GPIO_PIN_14, GPIO_PIN_SET);
	}

	if (dir2 == 0){
		HAL_GPIO_WritePin(DIR_2_GPIO_Port, GPIO_PIN_15, GPIO_PIN_SET);
	}
	else if (dir2 == 1){
		HAL_GPIO_WritePin(DIR_2_GPIO_Port, GPIO_PIN_15, GPIO_PIN_RESET);
	}
}

void motors_go(int speed)
{
	if (speed == 0) current_speed >= 49999? accelerate(49999) : decelerate(49999);

	else if (speed == 1) current_speed >= 39999? accelerate(39999) : decelerate(39999);

	else if (speed == 2) current_speed >= 29999? accelerate(29999) : decelerate(29999);

	else if (speed == 3) current_speed >= 19999? accelerate(19999) : decelerate(19999);

	else if (speed == 4) current_speed >= 9999? accelerate(9999) : decelerate(9999);
}

void accelerate(int final)
{
	int i;

	for(i = current_speed; i>=final; i-= 1000)
	{
		__HAL_TIM_SET_AUTORELOAD(&htim10, i);
		__HAL_TIM_SET_AUTORELOAD(&htim11, i);

		HAL_Delay(100);
	}

	current_speed = final;
}

void decelerate(int final)
{
	int i;

	for(i = current_speed; i<=final; i+= 1000)
	{
		__HAL_TIM_SET_AUTORELOAD(&htim10, i);
		__HAL_TIM_SET_AUTORELOAD(&htim11, i);

		HAL_Delay(200);
	}

	current_speed = final;
}
