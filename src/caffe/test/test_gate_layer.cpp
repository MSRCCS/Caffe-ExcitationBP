#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/gate_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class GateLayerTest : public GPUDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
	 GateLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 10, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
	//blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GateLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GateLayerTest, TestDtypesAndDevices);




TYPED_TEST(GateLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  this->blob_top_vec_.push_back(this->blob_top_);
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;

	InnerProductParameter* inner_product_param =
		layer_param.mutable_inner_product_param();
	inner_product_param->set_num_output(10);
	inner_product_param->mutable_weight_filler()->set_type("gaussian");

	this->blob_bottom_->mutable_cpu_data()[0] = 4;
	this->blob_bottom_->mutable_cpu_data()[1] = 2;
	this->blob_bottom_->mutable_cpu_data()[2] = 2;
	this->blob_bottom_->mutable_cpu_data()[3] = 3;
	//checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
	//	this->blob_top_vec_, -2);

	GateLayer<Dtype> layer(layer_param);
	//layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}


}  // namespace caffe
