#include <iostream>
#include <regex>
#include <string>
#include <vector>
#include <stack>
#include <fstream>
#include <map>
#include <math.h>
#include <net.h>
#include "prompt_slover.h"
#include "decoder_slover.h"
#include "diffusion_slover.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <time.h>
#include "getmem.h"
using namespace std;

int main()
{
	int step, seed;
	string positive_prompt, negative_prompt;

	// default setting
	step = 30;
	seed = 20230302;
	positive_prompt = "best quality, ultra high res, (photorealistic), (1 girl), thighhighs, (big chest), (upper body), (Kpop idol), (aegyo sal), (platinum blonde hair), ((puffy eyes)), looking at viewer, facing front, smiling, ((naked))";
	negative_prompt = "paintings, sketches, (worst quality), (low quality), (normal quality), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glan, ((cloth))";

	// parse the magic.txt
	ifstream magic;
	magic.open("magic.txt");
	if (!magic) {
		cout << "can not find magic.txt, using the default setting" << endl;
	}
	else {
		string content = "";
		int i = 0;
		for (i = 0; i < 4; i++) {
			if (getline(magic, content)) {
				switch (i)
				{
				case 0:step = stoi(content);
				case 1:seed = stoi(content);
				case 2:positive_prompt = content;
				case 3:negative_prompt = content;
				default:break;
				}
			}
			else {
				break;
			}
		}
		if (i != 4) {
			cout << "magic.txt has wrong format, please fix it" << endl;
			return 0;
		}

	}
	if (seed == 0) {
		seed = (unsigned)time(NULL);
	}
	magic.close();

	// stable diffusion
	cout << "----------------[init]--------------------";
	PromptSlover prompt_slover;
	DiffusionSlover diffusion_slover;
	DecodeSlover decode_slover;
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[prompt]------------------";
	ncnn::Mat cond = prompt_slover.get_conditioning(positive_prompt);
	ncnn::Mat uncond = prompt_slover.get_conditioning(negative_prompt);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[diffusion]---------------" << endl;
	ncnn::Mat sample = diffusion_slover.sampler(seed, step, cond, uncond);
	cout << "----------------[diffusion]---------------";
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[decode]------------------";
	ncnn::Mat x_samples_ddim = decode_slover.decode(sample);
	printf(" %.2lfG / %.2lfG\n", getCurrentRSS() / 1024.0 / 1024.0 / 1024.0, getPeakRSS() / 1024.0 / 1024.0 / 1024.0);

	cout << "----------------[save]--------------------" << endl;
	cv::Mat image(512, 512, CV_8UC3);
	x_samples_ddim.to_pixels(image.data, ncnn::Mat::PIXEL_RGB2BGR);
	cv::imwrite("result_" + to_string(step) + "_" + to_string(seed) + ".png", image);

	cout << "----------------[close]-------------------" << endl;
	
	return 0;
}
