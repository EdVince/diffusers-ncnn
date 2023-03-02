#include "diffusion_slover.h"

DiffusionSlover::DiffusionSlover()
{
	net.opt.use_vulkan_compute = false;
	net.opt.lightmode = true;
	net.opt.use_winograd_convolution = false;
	net.opt.use_sgemm_convolution = false;
	net.opt.use_fp16_packed = true;
	net.opt.use_fp16_storage = true;
	net.opt.use_fp16_arithmetic = true;
	net.opt.use_packing_layout = true;

	net.load_param("assets/unet.ncnn.param");
	net.load_model("assets/unet.ncnn.bin");

	ifstream in("assets/log_sigmas.bin", ios::in | ios::binary);
	in.read((char*)&log_sigmas, sizeof log_sigmas);
	in.close();
}

ncnn::Mat DiffusionSlover::randn_4(int seed)
{
	cv::Mat cv_x(cv::Size(64, 64), CV_32FC4);
	cv::RNG rng(seed);
	rng.fill(cv_x, cv::RNG::NORMAL, 0, 1);
	ncnn::Mat x_mat(64, 64, 4, (void*)cv_x.data);
	return x_mat.clone();
}

ncnn::Mat DiffusionSlover::CFGDenoiser_CompVisDenoiser(ncnn::Mat& input, float sigma, ncnn::Mat cond, ncnn::Mat uncond)
{
	// get_scalings
	float c_out = -1.0 * sigma;
	float c_in = 1.0 / sqrt(sigma * sigma + 1);

	// sigma_to_t
	float log_sigma = log(sigma);
	vector<float> dists(1000);
	for (int i = 0; i < 1000; i++) {
		if (log_sigma - log_sigmas[i] >= 0)
			dists[i] = 1;
		else
			dists[i] = 0;
		if (i == 0) continue;
		dists[i] += dists[i - 1];
	}
	int low_idx = min(int(max_element(dists.begin(), dists.end()) - dists.begin()), 1000 - 2);
	int high_idx = low_idx + 1;
	float low = log_sigmas[low_idx];
	float high = log_sigmas[high_idx];
	float w = (low - log_sigma) / (low - high);
	w = max(0.f, min(1.f, w));
	float t = (1 - w) * low_idx + w * high_idx;

	ncnn::Mat t_mat(1);
	t_mat[0] = t;

	// c_in
	ncnn::Mat in0_c_in(64, 64, 4);
	{
		for (int c = 0; c < 4; c++)
		{
			float* in = input.channel(c);
			float* ou = in0_c_in.channel(c);
			for (int hw = 0; hw < 64 * 64; hw++)
			{
				*ou = *in * c_in;
				ou++;
				in++;
			}
		}
	}

	ncnn::Mat denoised_cond;
	{
		ncnn::Extractor ex = net.create_extractor();
		ex.set_light_mode(true);
		ex.input("in0", in0_c_in);
		ex.input("in1", t_mat);
		ex.input("in2", cond);
		ex.extract("out0", denoised_cond);
	}

	// c_out
	{
		for (int c = 0; c < 4; c++)
		{
			float* in = input.channel(c);
			float* ou = denoised_cond.channel(c);
			for (int hw = 0; hw < 64 * 64; hw++)
			{
				*ou = *in + *ou * c_out;
				ou++;
				in++;
			}
		}
	}

	ncnn::Mat denoised_uncond;
	{
		ncnn::Extractor ex = net.create_extractor();
		ex.set_light_mode(true);
		ex.input("in0", in0_c_in);
		ex.input("in1", t_mat);
		ex.input("in2", uncond);
		ex.extract("out0", denoised_uncond);
	}

	// c_out
	{
		for (int c = 0; c < 4; c++)
		{
			float* in = input.channel(c);
			float* ou = denoised_uncond.channel(c);
			for (int hw = 0; hw < 64 * 64; hw++)
			{
				*ou = *in + *ou * c_out;
				ou++;
				in++;
			}
		}
	}

	for (int c = 0; c < 4; c++) {
		float* u_ptr = denoised_uncond.channel(c);
		float* c_ptr = denoised_cond.channel(c);
		for (int hw = 0; hw < 64 * 64; hw++) {
			(*u_ptr) = (*u_ptr) + 7 * ((*c_ptr) - (*u_ptr));
			u_ptr++;
			c_ptr++;
		}
	}

	return denoised_uncond;
}

ncnn::Mat DiffusionSlover::sampler(int seed, int step, ncnn::Mat& c, ncnn::Mat& uc)
{
	ncnn::Mat x_mat = randn_4(seed % 1000);

	// t_to_sigma
	vector<float> sigma(step);
	float delta = 0.0 - 999.0 / (step - 1);
	for (int i = 0; i < step; i++) {
		float t = 999.0 + i * delta;
		int low_idx = floor(t);
		int high_idx = ceil(t);
		float w = t - low_idx;
		sigma[i] = exp((1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]);
	}
	sigma.push_back(0.f);

	float _norm_[4] = { sigma[0], sigma[0], sigma[0], sigma[0] };
	x_mat.substract_mean_normalize(0, _norm_);

	
	// euler ancestral
	{
		for (int i = 0; i < sigma.size() - 1; i++) {
			cout << "step:" << i << "\t\t";

			double t1 = ncnn::get_current_time();
			ncnn::Mat denoised = CFGDenoiser_CompVisDenoiser(x_mat, sigma[i], c, uc);
			double t2 = ncnn::get_current_time();
			cout << t2 - t1 << "ms" << endl;

			float sigma_up = min(sigma[i + 1], sqrt(sigma[i + 1] * sigma[i + 1] * (sigma[i] * sigma[i] - sigma[i + 1] * sigma[i + 1]) / (sigma[i] * sigma[i])));
			float sigma_down = sqrt(sigma[i + 1] * sigma[i + 1] - sigma_up * sigma_up);

			srand(time(NULL) + i);
			ncnn::Mat randn = randn_4(rand() % 1000);
			for (int c = 0; c < 4; c++) {
				float* x_ptr = x_mat.channel(c);
				float* d_ptr = denoised.channel(c);
				float* r_ptr = randn.channel(c);
				for (int hw = 0; hw < 64 * 64; hw++) {
					*x_ptr = *x_ptr + ((*x_ptr - *d_ptr) / sigma[i]) * (sigma_down - sigma[i]) + *r_ptr * sigma_up;
					x_ptr++;
					d_ptr++;
					r_ptr++;
				}
			}
		}
	}
	
	
	/*
	// DPM++ 2M Karras
	ncnn::Mat old_denoised;
	{
		for (int i = 0; i < sigma.size() - 1; i++) {
			cout << "step:" << i << "\t\t";

			double t1 = ncnn::get_current_time();
			ncnn::Mat denoised = CFGDenoiser_CompVisDenoiser(x_mat, sigma[i], c, uc);
			double t2 = ncnn::get_current_time();
			cout << t2 - t1 << "ms" << endl;

			float sigma_curt = sigma[i];
			float sigma_next = sigma[i + 1];
			float tt = -1.0 * log(sigma_curt);
			float tt_next = -1.0 * log(sigma_next);
			float hh = tt_next - tt;
			if (old_denoised.empty() || sigma_next == 0)
			{
				for (int c = 0; c < 4; c++) {
					float* x_ptr = x_mat.channel(c);
					float* d_ptr = denoised.channel(c);
					for (int hw = 0; hw < 64 * 64; hw++) {
						*x_ptr = (sigma_next / sigma_curt) * *x_ptr - (exp(-hh) - 1) * *d_ptr;
						x_ptr++;
						d_ptr++;
					}
				}
			}
			else
			{
				float hh_last = -1.0 * log(sigma[i - 1]);
				float r = hh_last / hh;
				for (int c = 0; c < 4; c++) {
					float* x_ptr = x_mat.channel(c);
					float* d_ptr = denoised.channel(c);
					float* od_ptr = old_denoised.channel(c);
					for (int hw = 0; hw < 64 * 64; hw++) {
						*x_ptr = (sigma_next / sigma_curt) * *x_ptr - (exp(-hh) - 1) * ((1 + 1 / (2 * r)) * *d_ptr - (1 / (2 * r)) * *od_ptr);
						x_ptr++;
						d_ptr++;
						od_ptr++;
					}
				}
			}
			old_denoised.clone_from(denoised);
		}
	}
	*/
	

	ncnn::Mat fuck_x;
	fuck_x.clone_from(x_mat);
	return fuck_x;
}
