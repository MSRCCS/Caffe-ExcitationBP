#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/unpooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void UnpoolingForward(const int channels_, const int height_, const int width_,
    const Dtype* bottom_data,const Dtype* mask_data, Dtype* top_data, int bdos, int mdos, int topos) {
    CUDA_KERNEL_LOOP(c, channels_) {
      for (int ph = 0; ph < height_; ++ph) {
        for (int pw = 0; pw < width_; ++pw) {
          const int index = ph * width_ + pw;
          top_data[static_cast<int>(mask_data[index+c*mdos]) + c * topos] = bottom_data[index+c*bdos];
        }
      }
    }
}


template <typename Dtype>
void UnPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* mask_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int top_count = top[0]->count();

  const int count = bottom[0]->count();
  caffe_gpu_set(top_count, Dtype(0.), top_data);

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
	UnpoolingForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->num() * channels_), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->num() * channels_, height_, width_, bottom_data, mask_data, top_data, bottom[0]->offset(0, 1),bottom[1]->offset(0, 1),top[0]->offset(0, 1));

   
    break;
  case PoolingParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
	
  }
  CUDA_POST_KERNEL_CHECK;
}



template <typename Dtype>
__global__ void UnpoolingBackward(const int channels_, const int height_, const int width_,
    Dtype* bottom_diff,const Dtype* mask_data, const Dtype* top_diff, int bdos, int mdos, int topos) {
    CUDA_KERNEL_LOOP(c, channels_) {
      for (int ph = 0; ph < height_; ++ph) {
        for (int pw = 0; pw < width_; ++pw) {
          const int index = ph * width_ + pw;
		  bottom_diff[index+c*bdos] += top_diff[static_cast<int>(mask_data[index+c*mdos]) + c * topos];
        }
      }
    }
}




template <typename Dtype>
void UnPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	  if (!propagate_down[0]) {
		return;
	  }
	  
	  

	  const Dtype* top_diff = top[0]->gpu_diff();
	  const Dtype* mask_data = bottom[1]->gpu_data();
	  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	  const int top_count = top[0]->count();

	  const int count = bottom[0]->count();
	  caffe_gpu_set(count, Dtype(0.), bottom_diff);


	  switch (this->layer_param_.pooling_param().pool()) {
	  case PoolingParameter_PoolMethod_MAX:
		// NOLINT_NEXT_LINE(whitespace/operators)
		UnpoolingBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->num() * channels_), CAFFE_CUDA_NUM_THREADS>>>(
		  bottom[0]->num() * channels_, height_, width_, bottom_diff, mask_data, top_diff, bottom[0]->offset(0, 1),bottom[1]->offset(0, 1),top[0]->offset(0, 1));
		break;
	  case PoolingParameter_PoolMethod_AVE:
		// NOLINT_NEXT_LINE(whitespace/operators)    
		NOT_IMPLEMENTED;
		break;
	  case PoolingParameter_PoolMethod_STOCHASTIC:
		// NOLINT_NEXT_LINE(whitespace/operators)
		NOT_IMPLEMENTED;
		break;
	  default:
		LOG(FATAL) << "Unknown pooling method.";
	  }
	  CUDA_POST_KERNEL_CHECK;

}
INSTANTIATE_LAYER_GPU_FUNCS(UnPoolingLayer);

}  // namespace caffe
