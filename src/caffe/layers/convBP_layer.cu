#include <vector>

#include "caffe/layers/convBP_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void pos_kernel_convBP(const int n, const Dtype* a, Dtype* b) {
  CUDA_KERNEL_LOOP(index, n) {
    if (a[index] > 0)
      b[index] = a[index];
  }
}

template <typename Dtype>
__global__ void div_r_kernel_convBP(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    if (b[index] != 0)
      y[index] = a[index] / b[index];
    else
      y[index] = 0;
  }
}


template <typename Dtype>
void ConvolutionBPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<int> weight_shape = this->blobs_[0]->shape();
  vector<int> top_diff_shape = top[0]->shape(); 
  vector<int> bottom_diff_shape = bottom[0]->shape(); 
  vector<int> bottom_org_shape = bottom[1]->shape(); 





  bool bMatch = weight_shape[0]*weight_shape[2]*weight_shape[3]==top_diff_shape[1]*top_diff_shape[2]*top_diff_shape[3] 
       && weight_shape[0]*weight_shape[2]*weight_shape[3]==bottom_org_shape[1]*bottom_org_shape[2]*bottom_org_shape[3] // Match of original bottom
             && weight_shape[1]==bottom_diff_shape[1]; // Match of the diff to be propaged down

  // ToDo: Move the monitor in once code is stable
  if ( false )
  {
     LOG(INFO)<<"Name of layer: " << this->layer_param_.name();
     LOG(INFO)<<"Shape of W_data:" << this->blobs_[0]->shape_string();
     LOG(INFO)<<"Shape of top: "<<top[0]->shape_string();
     for ( int s=0; s<(int)bottom.size(); s++ )
     {
        LOG(INFO)<<"Shape of bottom " << s <<": "<<bottom[s]->shape_string();
     }
  };
  if ( !bMatch ) 
  {
     //LOG(FATAL)<<"Mismatched shape for the ConvBP layer";
     LOG(WARNING)<<"Mismatched shape for the ConvBP layer";
  }
  // get the new weight W+
  const Dtype* W_data = this->blobs_[0]->gpu_data();
  Blob<Dtype> W_plus(this->blobs_[0]->shape());
  Dtype* W_plus_data = W_plus.mutable_gpu_data();
  caffe_gpu_set<Dtype>(W_plus.count(), Dtype(0), W_plus_data);
  pos_kernel_convBP<Dtype><<<CAFFE_GET_BLOCKS(W_plus.count()), CAFFE_CUDA_NUM_THREADS>>>(
        W_plus.count(), W_data, W_plus_data);

  LOG(INFO)<<"Copy W_data to W_plus";
  Blob<Dtype> NN(bottom[0]->shape());
  Dtype* NN_data = NN.mutable_gpu_data();
  
  //LOG(INFO)<<"This Num is: "<< this->num_<<" Group is: "<< this-> group_;
  //LOG(INFO)<<"Conv out channels is: "<< this->conv_out_channels_;
  //LOG(INFO)<<"Conv out spatial dimension is: "<< this->conv_out_spatial_dim_; 
  //LOG(INFO)<<"Kernel dimension is: "<< this->kernel_dim_; 


  // The following name follows the convention of the mirrored forward path 
  // on the classification path.
  // compute the normalization factor by forwardpassing using W+
  const Dtype* bottom_data = bottom[1]->gpu_data();
  for (int n = 0; n < this->num_; ++n) {
    this->forward_gpu_gemm(bottom_data + n * this->top_dim_, W_plus_data,
      NN_data + n * this->bottom_dim_);
  }




  //this->print_vector(bottom_data, bottom[1]->count() );
              //this->print_vector( W_plus_data, this->blobs_[0]->count() );
              //this->print_vector( NN_data, bottom[0]->count() );
  // do normalization

  const Dtype* top_diff = bottom[0]->gpu_data();
  div_r_kernel_convBP<Dtype><<<CAFFE_GET_BLOCKS(NN.count()), CAFFE_CUDA_NUM_THREADS>>>(
        NN.count(), top_diff, NN_data, NN_data);



              //LOG(INFO)<<"Done Compute normalization";
              //this->print_vector( NN_data, bottom[0]->count() );
  // do backward pass
  Dtype* bottom_diff = top[0]->mutable_gpu_data();
  for (int n = 0; n < this->num_; ++n) {
    this->backward_gpu_gemm(NN_data + n * this->bottom_dim_, W_plus_data,
      bottom_diff + n * this->top_dim_);
  }

              //LOG(INFO)<<"Done backward pass";
              //this->print_vector( bottom_diff, top[0]->count() );

  // multiply the bottom data
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, bottom_data, bottom_diff);

              //LOG(INFO)<<"Multiply bottom data";
              //this->print_vector( bottom_diff, top[0]->count() );


















}

template <typename Dtype>
void ConvolutionBPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void pos_kernel(const int n, const Dtype* a, Dtype* b) {
  CUDA_KERNEL_LOOP(index, n) {
    if (a[index] > 0)
      b[index] = a[index];
  }
}

template <typename Dtype>
__global__ void div_r_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    if (b[index] != 0)
      y[index] = a[index] / b[index];
    else
      y[index] = 0;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionBPLayer);

}  // namespace caffe
