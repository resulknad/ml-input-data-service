// Modified from core/ops/stateless_random_ops.cc and random_ops.cc and stateless_random_ops_v2.cc

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rng_alg.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// Removing the necessity of seed from the Ops

static Status DeterministicOpsShape(InferenceContext* c) {
  /*
  // Check seed shape and transfer the shape to the shapehandle seed
  ShapeHandle seed;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &seed));
  DimensionHandle unused;
  // check the dimension value is 2 or not
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(seed, 0), 2, &unused));
  */
  ShapeHandle key;
  ShapeHandle counter;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &key));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &counter));
  shape_inference::ShapeHandle unused_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused_shape));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(key, 0), RNG_KEY_SIZE, &unused));

  // Set output shape
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
  c->set_output(0, out);
  return Status::OK();
}
// No op level seed defined. global_seed is the attribute to capture the tf.ramdom.set_seed
// #define REGISTER_DETERMINISTIC_OP(name)                           \
//   REGISTER_OP(name)                                           \
//       .Input("shape: T")                                      \
//       .Output("output: dtype")                                \
//       .Attr("dtype: {half,bfloat16,float,double} = DT_FLOAT") \
//       .Attr("T: {int32, int64} = DT_INT32")                   \
//       .Attr("global_seed: {int32, int64} = DT_INT64")           

//Fixing the global seed as an attribute for the OP when it registers to the graph

#define REGISTER_DETERMINISTIC_OP(name)                           \
  REGISTER_OP(name)                                           \
      .Input("shape: Tshape")                                 \
      .Input("key: uint64")                                   \
      .Input("counter: uint64")                               \
      .Input("alg: int32")                                    \
      .Output("output: dtype")                                \
      .Attr("dtype: {half,bfloat16,float,double} = DT_FLOAT") \
      .Attr("Tshape: {int32, int64} = DT_INT32")              \
      .Attr("global_seed: {int32, int64} = DT_INT64")         \
      .SetShapeFn(DeterministicOpsShape)

REGISTER_DETERMINISTIC_OP("DeterministicRandomUniform");
//REGISTER_DETERMINISTIC_OP("DeterministicRandomNormal");
//REGISTER_DETERMINISTIC_OP("DeterministicTruncatedNormal");

#undef REGISTER_DETERMINISTIC_OP

REGISTER_OP("DeterministicRandomUniformInt")
    .Input("shape: Tshape")
    .Input("key: uint64")
    .Input("counter: uint64")
    .Input("alg: int32")
    .Input("minval: dtype")
    .Input("maxval: dtype")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64, uint32, uint64}")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .Attr("global_seed: {int32, int64} = DT_INT64")         //added
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      Status s = c->WithRank(c->input(4), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "minval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(4)));
      }
      s = c->WithRank(c->input(5), 0, &unused);
      if (!s.ok()) {
        return errors::InvalidArgument(
            "maxval must be a scalar; got a tensor of shape ",
            c->DebugString(c->input(5)));
      }
      return DeterministicOpsShape(c);
    });

REGISTER_OP("DeterministicRandomUniformFullIntV2")
    .Input("shape: Tshape")
    .Input("key: uint64")
    .Input("counter: uint64")
    .Input("alg: int32")
    .Output("output: dtype")
    .Attr("dtype: {int32, int64, uint32, uint64} = DT_UINT64")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .Attr("global_seed: {int32, int64} = DT_INT64")         //added
    .SetShapeFn(DeterministicOpsShape);

    // TODO Support for other Deterministic ops and utils