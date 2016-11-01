#include <vector>

#include "caffe/layers/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GateForward(const int n, const Dtype* in,
    const Dtype* mask,
    Dtype* out, const int channels) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * mask[index % channels];
  }
}

template <typename Dtype>
void GateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();


	const Dtype* mask = this->blobs_[0]->gpu_data();
	GateForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		count, bottom_data, mask, top_data,bottom[0]->channels());
	CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
__global__ void GateBackward(const int n, const Dtype* in_diff,
    const Dtype* mask,  Dtype* out_diff, bool blob_diff, const int channels) {
  CUDA_KERNEL_LOOP(index, n) {
	if (!blob_diff){
		out_diff[index] = in_diff[index] * mask[index % channels];
	}
	else 
	{
		out_diff[index % channels] += in_diff[index] * mask[index];
	}
  }
}

template <typename Dtype>
void GateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (this->param_propagate_down_[0]) {
      const Dtype* bottom_data = bottom[0]->gpu_data();
	  Dtype* blob_diff = this->blobs_[0]->mutable_gpu_diff();
	  caffe_gpu_set(this->blobs_[0]->count(), Dtype(0.), blob_diff);
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)

        GateBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, bottom_data, blob_diff, true,bottom[0]->channels());
       
      CUDA_POST_KERNEL_CHECK;
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* mask = this->blobs_[0]->gpu_data();
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)

        GateBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask,  bottom_diff, false,bottom[0]->channels());
       
      CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GateLayer);

}  // namespace caffe
