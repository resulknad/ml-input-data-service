#include <queue>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_get_op.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace easl {
namespace {
  // Define constants here
  constexpr char kCacheLocation[] = 
      "/mnt/local/easl/dan-mlid/scripts/python/outputs/00000000.snapshot";
  constexpr char kPutOpDataset[] = "ServiceCacheGetDataset";
  constexpr char kOutputShapes[] = "output_shapes";
  constexpr char kOutputTypes[] = "output_types";

  NodeDef CreateGetOpNode(MutableGraphView* graph, NodeDef* input) {
    // TODO(DanGraur): Change the implementation here
    NodeDef get_op_node;

    // Give a unique name to the op
    graph_utils::SetUniqueGraphNodeName("get_op_dataset",
        graph->graph(), &get_op_node);

    // Set the node's operation and input.
    get_op_node.set_op(kPutOpDataset);

    NodeDef* location_node = graph_utils::AddScalarConstNode(kCacheLocation, 
        graph); 
    get_op_node.add_input(location_node->name());

    // FIXME(DanGraur): Finish the implementation of this

    // Copy over the relevant attributes from root of the prefix
    for (auto key : {kOutputShapes, kOutputTypes})
      graph_utils::CopyAttribute(key, *input, &get_op_node);

    return get_op_node;
  }
} // namespace

Status AddGetOp::OptimizeAndCollectStats(Cluster* cluster,
                                         const GrapplerItem& item,
                                         GraphDef* output,
                                         OptimizationStats* stats) {
  // TODO(DanGraur): Change the implementation here
  VLOG(1) << "In AddGetOp optimizer";
  *output = item.graph;
  MutableGraphView graph(output);

  // Define a filtering function which identifies target node
  auto is_target_node = [](const NodeDef* node) -> bool {
    return node->op() == "ModelDataset" && node->input_size() == 1;  
  };

  // Get the output of the graph
  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  // Find the first target op by applying BFS
  absl::flat_hash_set<std::string> visited;
  std::queue<NodeDef*> bfs_queue;
  bfs_queue.push(sink_node);
  NodeDef* target = nullptr;

  while (!bfs_queue.empty()) {
    NodeDef* current_node = bfs_queue.front();
    bfs_queue.pop();
    visited.insert(current_node->name());

    // TODO(DanGraur): Add logic here to skip certain nodes (e.g. control)

    // Check to see if this node is a target op
    if (is_target_node(current_node)) {
      target = current_node;
      break;
    }

    // Iterate throught the neighbors
    for (int i = 0; i < current_node->input_size(); ++i) {
      if (!visited.contains(current_node->input(i))) {
        int idx = graph_utils::FindGraphNodeWithName(current_node->input(i), 
            *output);
        NodeDef* neighbor_node = output->mutable_node(idx);
        bfs_queue.push(neighbor_node);
      }
    }
  }

  // We return if we found no target op
  if (!target) {
    VLOG(1) << "Could not find target";
    return Status::OK();
  }

  // Find the input of the target node
  NodeDef* target_input = graph_utils::GetInputNode(*target, graph);
  if(!target_input){
    return errors::Unknown("The dataset graph sink node does not have"
    "an input.");
  }
  
  // Create the put_op_node op node, then add it to the graph
  NodeDef put_op_node = CreatePutOpNode(&graph, target_input);

  // Copy over the relevant attributes
  (*target->mutable_input())[0] = put_op_node.name();
  graph_utils::CopyAttribute(kOutputTypes, put_op_node, target);

  // Add the node to the graph
  graph.AddNode(std::move(put_op_node));

  return Status::OK();
}

void AddGetOp::Feedback(Cluster* cluster, const GrapplerItem& item,
                              const GraphDef& optimize_output,
                              double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(AddGetOp, "add_get_op");

}  // namespace easl
}  // namespace grappler
}  // namespace tensorflow
