#include <algorithm>
#include <vector>

#include "caffe/layers/channel_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ChannelNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
}

template <typename Dtype>
void ChannelNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

}

template <typename Dtype>
void ChannelNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num = bottom[0]->shape(0);
  int channels = bottom[0]->shape(1);
  int innerNum = bottom[0]->count() / (num * channels);
  for (int n = 0; n < bottom[0]->num(); ++n) {
	  for (int c = 0; c < channels_; ++c) {
		  Dtype min = bottom_data[0];
		  Dtype max = bottom_data[0];
		  for (int index = 0; index < innerNum; ++index)
		  {
			  if (bottom_data[index] > max)
			  {
				  max = bottom_data[index];
			  }
			  if (bottom_data[index] < min)
			  {
				  min = bottom_data[index];
			  }
		  }

		  for (int index = 0; index < innerNum; ++index)
		  {
			  top_data[index] = (bottom_data[index] - min) / (max - min);
			  top_data[index] = top_data[index] > 0.5 ? (top_data[index] - 0.5) * 2:(0.5 - top_data[index]) * 2;
		  }

		  bottom_data += bottom[0]->offset(0, 1);
		  top_data += top[0]->offset(0, 1);
	  }

  }

    

}

template <typename Dtype>
void ChannelNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
	  caffe_copy(bottom[0]->count(), top[0]->cpu_diff(), bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(ChannelNormLayer);
#endif

INSTANTIATE_CLASS(ChannelNormLayer);
REGISTER_LAYER_CLASS(ChannelNorm);
}  // namespace caffe
