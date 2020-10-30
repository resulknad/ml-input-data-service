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
  constexpr char kFortyTwoDataset[] = "FortyTwoDataset";

}

Status AppendFortyTwo::OptimizeAndCollectStats(Cluster* cluster,
                                                    const GrapplerItem& item,
                                                    GraphDef* output,
                                                    OptimizationStats* stats) {
  *output = item.graph;
  
  MutableGraphView graph(output);

  // For now, just be a no-op, we'll try to print things from here.
  VLOG(1) << "Inside the append_forty_two optimization";

  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  NodeDef* forty_two_input = graph_utils::GetInputNode(sink_node, graph);
  // Should never ever be null. Use assert instead?
  // assert(forty_two_input);
  if(!forty_two_input){
    return errors::Unkown("The dataset graph sink node does not have"
    "an input.");
  }

  NodeDef forty_two_node;
  // Give a unique name to our forty_two node and store it for later use
  graph_utils::SetUniqueGraphNodeName("forty_two", output, &forty_two_node);
  std::string forty_two_node_name = node.name();
  // Set its operation and input.
  node.set_op(kFortyTwoDataset);
  node.add_input(forty_two_input->name());
  // Add the node to the graph.
  graph_utils::AddNode(std::move(forty_two_node));
  // Modify the input of the sink node to be the forty_two node.
  sink_node->set_input(forty_two_node_name);

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
