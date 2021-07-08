#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_APPEND_FORTY_TWO_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_APPEND_FORTY_TWO_H_

#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {
namespace easl {

// This optimization adds 'forty_two_dataset_op' somewhere random in the 
// input pipeline graph
class AppendFortyTwo : public TFDataOptimizerBase {
  public:
  AppendFortyTwo() = default;
  ~AppendFortyTwo() override = default;

  string name() const override { return "append_forty_two"; };

  bool UsesFunctionLibrary() const override { return false; };

  Status Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    // Ignore configuration
    return Status::OK();
  }

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;


};

} // namespace easl
} // namespace grappler
} // namespace tensorflow

#endif // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_APPEND_FORTY_TWO_H_