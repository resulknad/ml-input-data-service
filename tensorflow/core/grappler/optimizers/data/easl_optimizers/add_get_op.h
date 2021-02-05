#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ADD_GET_OP_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ADD_GET_OP_H_

#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {
namespace easl {

// Appends the cache Get op before the ModelDatasetOp node 
class AddGetOp : public TFDataOptimizerBase {
  public:
  AddGetOp() = default;
  ~AddGetOp() override = default;

  string name() const override { return "add_get_op"; };

  bool UsesFunctionLibrary() const override { return false; };

  Status Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    // Ignore configuration
    return Status::OK();
  }
  
  Status ApplyOptimization(MutableGraphView &graph, NodeDef *sink_node, 
                           GraphDef *output);

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

};

} // namespace easl
} // namespace grappler
} // namespace tensorflow

#endif // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ADD_GET_OP_H_