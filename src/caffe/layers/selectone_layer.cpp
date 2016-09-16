#include <algorithm>
#include <vector>

#include "caffe/layers/selectone_layer.hpp"

namespace caffe {

template <typename Dtype>
void SelectOneLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  int maxidx = 0;
  for (int i = 0; i < count; ++i) {
    if (bottom_data[i] > bottom_data[maxidx])
    {
      maxidx = i;
      top_data[i] = 0;
    }
  }
  top_data[maxidx] = bottom_data[maxidx];
}

template <typename Dtype>
void SelectOneLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    //const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i];
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SelectOneLayer);
#endif

INSTANTIATE_CLASS(SelectOneLayer);
REGISTER_LAYER_CLASS(SelectOne);
}  // namespace caffe
