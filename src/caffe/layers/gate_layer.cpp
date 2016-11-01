// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  top[0]->ReshapeLike(*bottom[0]);
  channels_ = bottom[0]->channels();
  nums_ = bottom[0]->num();

	this->blobs_.resize(1);

	// Initialize the weights
	vector<int> weight_shape(1);

	weight_shape[0] = channels_;
	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
	// fill the weights
	shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
		this->layer_param_.inner_product_param().weight_filler()));
	weight_filler->Fill(this->blobs_[0].get());
	// If necessary, intiialize and fill the bias term
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void GateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  channels_ = bottom[0]->channels();
  nums_ = bottom[0]->num();
  //this->blobs_[0].Reshape(bottom[0]->channels());

  //LOG(WARNING) << "============Reshape Gate Layer ===============";

  //this->blobs_.resize(1);

  //// Initialize the weights
  //vector<int> weight_shape(1);

  //weight_shape[0] = channels_;
  //this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  //// fill the weights
  //shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
  //this->layer_param_.inner_product_param().weight_filler()));
  //weight_filler->Fill(this->blobs_[0].get());
  //// If necessary, intiialize and fill the bias term
  //// parameter initialization
  //this->param_propagate_down_.resize(this->blobs_.size(), true);


}

template <typename Dtype>
void GateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* mask = this->blobs_[0]->cpu_data();

  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
	top_data[i] = bottom_data[i] * mask[i % channels_];
  }
}

template <typename Dtype>
void GateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	if (this->param_propagate_down_[0]) {
		const Dtype* top_diff = top[0]->cpu_diff();
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* blob_diff = this->blobs_[0]->mutable_cpu_diff();
		//caffe_cpu_set(channels_, Dtype(0.), blob_diff);
		const int count = bottom[0]->count();
		


	caffe_set(this->blobs_[0]->count(), static_cast<Dtype>(0), blob_diff);

		for (int i = 0; i < count; ++i) {
			blob_diff[i % channels_] += top_diff[i] * bottom_data[i];
		}
	}
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* mask = this->blobs_[0]->cpu_data();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
		bottom_diff[i] = top_diff[i] * mask[i % channels_];
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GateLayer);
#endif

INSTANTIATE_CLASS(GateLayer);
REGISTER_LAYER_CLASS(Gate);

}  // namespace caffe
