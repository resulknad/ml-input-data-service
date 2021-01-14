#include <queue>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_put_op.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace grappler {
namespace easl {
namespace {
  // Define constants here
  constexpr char kCacheLocation[] = "./outputs/";
  constexpr char kPutOpDataset[] = "ServiceCachePutDataset";
  constexpr char kOutputShapes[] = "output_shapes";
  constexpr char kOutputTypes[] = "output_types";
  //constexpr char kTargetNode[] = "ModelDataset";
  constexpr char kTargetNode[] = "ParallelMapDataset";
  constexpr int kTargetInputSize = 1;

  NodeDef CreatePutOpNode(MutableGraphView* graph, NodeDef* input) {
    NodeDef put_op_node;

    // Give a unique name to the op
    graph_utils::SetUniqueGraphNodeName("put_op_dataset",
        graph->graph(), &put_op_node);

    // Set the node's operation and inputs.
    put_op_node.set_op(kPutOpDataset);
    put_op_node.add_input(input->name());
    NodeDef* location_node = graph_utils::AddScalarConstNode<StringPiece>(
        kCacheLocation, graph); 
    put_op_node.add_input(location_node->name());

    // Copy over the relevant attributes from root of the prefix
    for (auto key : {kOutputShapes, kOutputTypes})
      graph_utils::CopyAttribute(key, *input, &put_op_node);

    return put_op_node;
  }
} // namespace

Status AddPutOp::ApplyOptimization(MutableGraphView &graph, NodeDef *sink_node,
                                   GraphDef *output) {
  VLOG(1) << "In AddPutOp optimizer";

  // Define a filtering function which identifies target node
  auto is_target_node = [](const NodeDef* node) -> bool {
    return node->op() == kTargetNode && node->input_size() == kTargetInputSize;
  };

  // Find the first target op by applying BFS
  absl::flat_hash_set<std::string> visited;
  std::queue<NodeDef*> bfs_queue;
  bfs_queue.push(sink_node);
  NodeDef* target = nullptr;

  while (!bfs_queue.empty()) {
    NodeDef* current_node = bfs_queue.front();
    bfs_queue.pop();
    visited.insert(current_node->name());

    VLOG(1) << "@ current_node: " << current_node->op();

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
    VLOG(1) << "Could not find target " << kTargetNode;
    return Status::OK();
  }

  // Find the input of the target node
  NodeDef* target_input = graph_utils::GetInputNode(*target, graph);
  if(!target_input){
    return errors::Unknown("The target has no inputs.");
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

Status AddPutOp::OptimizeAndCollectStats(Cluster* cluster,
                                         const GrapplerItem& item,
                                         GraphDef* output,
                                         OptimizationStats* stats) {
  // Initializations
  *output = item.graph;
  MutableGraphView graph(output);

  // Get the sink node
  NodeDef* sink_node;
  TF_RETURN_IF_ERROR(graph_utils::GetFetchNode(graph, item, &sink_node));

  // Apply the transformation
  return ApplyOptimization(graph, sink_node, output);
}

void AddPutOp::Feedback(Cluster* cluster, const GrapplerItem& item,
                              const GraphDef& optimize_output,
                              double result) {
  // no-op
}

REGISTER_GRAPH_OPTIMIZER_AS(AddPutOp, "add_put_op");

}  // namespace easl
}  // namespace grappler
}  // namespace tensorflow
