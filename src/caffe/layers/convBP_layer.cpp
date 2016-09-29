#include <vector>

#include "caffe/layers/convBP_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionBPLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    //const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
    //    / stride_data[i] + 1;

    //hongzl: output dim for convBP layer is different from the typical conv layer
    // use the some setup with deconv layer
    const int output_dim = stride_data[i] * (input_dim - 1)
        + kernel_extent - 2 * pad_data[i];
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionBPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  const Dtype* W_data = this->blobs_[0]->cpu_data();
  Blob<Dtype> W_plus(this->blobs_[0]->shape());
  Dtype* W_plus_data = W_plus.mutable_cpu_data();
  for (int i = 0; i < W_plus.count(); ++i) {
	  W_plus_data[i] = std::max(W_data[i], Dtype(0));
  }

  //LOG(INFO)<<"Copy W_data to W_plus";
  Blob<Dtype> NN(bottom[0]->shape());
  Dtype* NN_data = NN.mutable_cpu_data();
  
  //LOG(INFO)<<"This Num is: "<< this->num_<<" Group is: "<< this-> group_;
  //LOG(INFO)<<"Conv out channels is: "<< this->conv_out_channels_;
  //LOG(INFO)<<"Conv out spatial dimension is: "<< this->conv_out_spatial_dim_; 
  //LOG(INFO)<<"Kernel dimension is: "<< this->kernel_dim_; 


  {
         {  
                  // The following name follows the convention of the mirrored forward path 
                  // on the classification path.
		  // compute the normalization factor by forwardpassing using W+
		  const Dtype* bottom_data = bottom[1]->cpu_data();
		  for (int n = 0; n < this->num_; ++n) {
			  this->forward_cpu_gemm(bottom_data + n * this->top_dim_, W_plus_data,
				  NN_data + n * this->bottom_dim_);
		  }
		  //this->print_vector(bottom_data, bottom[1]->count() );
                  //this->print_vector( W_plus_data, this->blobs_[0]->count() );
                  //this->print_vector( NN_data, bottom[0]->count() );
		  // do normalization
		  const Dtype* top_diff = bottom[0]->cpu_data();
		  for (int j = 0; j < NN.count(); ++j) {
			  NN_data[j] = NN_data[j] == Dtype(0) ? Dtype(0) : (top_diff[j] / NN_data[j]);
		  }

                  //LOG(INFO)<<"Done Compute normalization";
                  //this->print_vector( NN_data, bottom[0]->count() );
		  // do backward pass
		  Dtype* bottom_diff = top[0]->mutable_cpu_data();
		  for (int n = 0; n < this->num_; ++n) {
			  this->backward_cpu_gemm(NN_data + n * this->bottom_dim_, W_plus_data,
				  bottom_diff + n * this->top_dim_);
		  }

                  //LOG(INFO)<<"Done backward pass";
                  //this->print_vector( bottom_diff, top[0]->count() );

		  // multiply the bottom data
		  caffe_mul<Dtype>(top[0]->count(), bottom_diff, bottom_data, bottom_diff);

                  //LOG(INFO)<<"Multiply bottom data";
                  //this->print_vector( bottom_diff, top[0]->count() );
	  }
  }
}

template <typename Dtype>
void ConvolutionBPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  NOT_IMPLEMENTED;

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}
    

#ifdef CPU_ONLY
STUB_GPU(ConvolutionBPLayer);
#endif

INSTANTIATE_CLASS(ConvolutionBPLayer);
REGISTER_LAYER_CLASS(ConvolutionBP);
}  // namespace caffe
