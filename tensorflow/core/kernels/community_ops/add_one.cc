#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;


class AddOneOp : public OpKernel {
  public:
    explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Get the input
      const Tensor& input = context->input(0);

      //Changing then implementation for the sake of rebuilding the thing..
      const Tensor& input2 = context->input(0);


      auto flat_input = input.flat<int32>();

      // Create the output and alocate memory
      Tensor* output = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
      auto flat_output = output->flat<int32>();

      // Add one to each entry
      for (int i = 0; i < flat_input.size(); ++i)
        flat_output(i) = flat_input(i) + 1;
    }
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_CPU), AddOneOp);
