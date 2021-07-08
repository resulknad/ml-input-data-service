#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ADD_PUT_OP_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ADD_PUT_OP_H_

#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {
namespace easl {

// Appends the cache Put op before the ModelDatasetOp node 
class AddPutOp : public TFDataOptimizerBase {
  public:
  AddPutOp() = default;
  ~AddPutOp() override = default;

  string name() const override { return "add_put_op"; };

  bool UsesFunctionLibrary() const override { return false; };

  Status Init(
    const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    config_ = *config;
    return Status::OK();
  }
  
  Status ApplyOptimization(MutableGraphView &graph, NodeDef *sink_node, 
                           GraphDef *output);

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

 private:
  NodeDef CreatePutOpNode(MutableGraphView * graph, NodeDef* input);

  tensorflow::RewriterConfig_CustomGraphOptimizer config_;

};

} // namespace easl
} // namespace grappler
} // namespace tensorflow

#endif // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ADD_PUT_OP_H_