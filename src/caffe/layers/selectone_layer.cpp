#include <algorithm>
#include <vector>

#include "caffe/layers/selectone_layer.hpp"

namespace caffe {


template <typename Dtype>
void SelectOneLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count(1);
  if (bottom.size()==1)
  {
    for (int n = 0; n < bottom[0]->num(); ++n) 
    {
      int maxidx = 0;
      for (int i = 0; i < count; ++i) 
      {
        if (bottom_data[i] > bottom_data[maxidx])
        {
          maxidx = i;
          top_data[i] = 0;
        }
      }
      top_data[maxidx] = 1;
      bottom_data += count;
      top_data += count;
      //top_data[maxidx] = bottom_data[maxidx];
    }
  }
  else
  {
    const Dtype* label_data = bottom[1]->cpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) 
    {
      // TODO: check why caffe_memset doesn't work in multiple iterations
      // the memset does not reset the memory in each iteration.
      //caffe_memset(top[0]->count(),0,top_data);
      for (int i = 0; i < count; ++i) 
      {
        top_data[i] = 0;
      }
      top_data[static_cast<int>(label_data[n])] = 1;
      bottom_data += count;
      top_data += count;   
    }
  }
  
  
}

template <typename Dtype>
void SelectOneLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
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
