#include "tensorflow/core/grappler/optimizers/data/append_forty_two.h"

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace {
  // Define constants here
}

Status AppendFortyTwo::OptimizeAndCollectStats(Cluster* cluster,
                                                    const GrapplerItem& item,
                                                    GraphDef* output,
                                                    OptimizationStats* stats) {
  *output = item.graph;
  
  MutableGraphView graph(output);

  // For now, just be a no-op, we'll try to print things from here.
  std::printf("Inside AppendFortyTwoOptimizer");

  // Hopefully at some point
  return Status::OK();
}

void AppendFortyTwo::Feedback(Cluster* cluster, const GrapplerItem& item,
                              const GraphDef& optimize_output,
                              double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(AppendFortyTwo, "append_forty_two");

}  // namespace grappler
}  // namespace tensorflow
