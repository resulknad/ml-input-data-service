#include <queue>

#include "tensorflow/core/data/service/easl/cache_utils.h"

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_put_op.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_get_op.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/graph_view.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/data/service/easl/cache_model.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_put_op_at_marker.h"
#include "tensorflow/core/grappler/optimizers/data/easl_optimizers/add_get_op_at_marker.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace cache_utils {

namespace {
  // Represents an offset which is subtracted from the non-rounded up worker count
  // This offset tries to avoid cases where a value such as 4.02 provisions 
  // 5 workers and not 4, as woul be ideal
  double worker_count_alpha_ = 0.1;
}

Status DoBFS(NodeDef* sink_node, GraphDef& graph_def, string prefix) {
  absl::flat_hash_set<std::string> visited;
  std::queue<NodeDef*> bfs_queue;
  bfs_queue.push(sink_node);

  VLOG(0) << "(" << prefix << ") BFS @ current_node: "
          << "Root --> " << sink_node->op();

  while (!bfs_queue.empty()) {
    NodeDef* current_node = bfs_queue.front();
    bfs_queue.pop();
    visited.insert(current_node->name());

    // Iterate throught the neighbors
    for (int i = 0; i < current_node->input_size(); ++i) {
      if (!visited.contains(current_node->input(i))) {
        int idx = tensorflow::grappler::graph_utils::FindGraphNodeWithName(
            current_node->input(i), graph_def);
        NodeDef* neighbor_node = graph_def.mutable_node(idx);
        bfs_queue.push(neighbor_node);

        VLOG(0) << "(" << prefix << ") BFS @ current_node: "
                << SummarizeNodeDef(*current_node) << " --> "
                << SummarizeNodeDef(*neighbor_node);
      }
    }
  }

  return Status::OK();
}

std::string DatasetPutKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_put");
}

std::string DatasetPutSourceKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_put_source");
}

std::string DatasetGetKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_get");
}

std::string DatasetGetSourceKey(const int64 id, const uint64 fingerprint) {
  return absl::StrCat("id_", id, "_fp_", fingerprint, "_get_source");
}

std::string DatasetKey(
    const int64 id, const uint64 fingerprint, const std::string& job_type){
  if(job_type=="COMPUTE"){
    return absl::StrCat("id_", id, "_fp_", fingerprint);
  } else if (job_type=="GET"){
    return DatasetGetKey(id, fingerprint);
  } else if (job_type=="PUT"){
    return DatasetPutKey(id, fingerprint);
  } else if (job_type=="GET_SOURCE") {
    return DatasetGetSourceKey(id, fingerprint);
  } else if (job_type=="PUT_SOURCE") {
    return DatasetPutSourceKey(id, fingerprint);
  }
  return "";
}

// TODO (damien-aymon) deprecated, left here for reference.
/*
Status DatasetKey(const ::tensorflow::data::easl::CacheState& cache_state,
                  const int64 dataset_id,
                  const uint64 fingerprint,
                  const std::string& worker_address,
                  const int64 task_id,
                  std::string& dataset_key){
  if(cache_state.IsDatasetCached(fingerprint, worker_address)){
    dataset_key =
        absl::StrCat("id_", dataset_id, "_fp_", fingerprint, "_get");
    VLOG(0) << "Use get dataset for fingerprint " << fingerprint
            << " at worker " << worker_address;
    return Status::OK();
  }

  int64 caching_task;
  TF_RETURN_IF_ERROR(cache_state.GetCachingTaskId(
      fingerprint, worker_address, caching_task));
  if(caching_task == task_id) {
    dataset_key =
        absl::StrCat("id_", dataset_id, "_fp_", fingerprint, "_put");
    VLOG(0) << "Use put dataset for fingerprint " << fingerprint
            << " at worker " << worker_address;
    return Status::OK();
  }

  dataset_key =
      absl::StrCat("id_", dataset_id, "_fp_", fingerprint);
  VLOG(0) << "Use standard dataset for fingerprint " << fingerprint
          << " at worker " << worker_address;
  return Status::OK();
}*/

Status DetermineJobType(const experimental::DispatcherConfig& dispatcher_config,
                     ::tensorflow::data::CacheState& cache_state,
                     const ::tensorflow::data::easl::MetadataStore& metadata_store,
                     const uint64 fingerprint,
                     const std::string& dataset_key,
                     const int64 job_id,
                     std::string& job_type) {
  // First check if we should use a "fixed" cache policy:
  // 1 == EASL
  // 2 == compute 
  // 3 == full cache(put, then get from 2nd epoch)
  // 4 == source cache(put, then get from 2nd epoch) 
  // Compute -------------------------------------------------------------------
  if(dispatcher_config.cache_policy() == 2){
    job_type = "COMPUTE";
    return Status::OK();
  // Caching -------------------------------------------------------------------
  } else if(dispatcher_config.cache_policy() == 3){
    if(cache_state.IsDatasetCached(fingerprint)){
      job_type = "GET";
    } else {
      job_type = "PUT";
    }
    return Status::OK();
  } else if (dispatcher_config.cache_policy() == 30) {
    job_type = "PUT";
    return Status::OK();
  } else if (dispatcher_config.cache_policy() == 31) {
    job_type = "GET";
    return Status::OK();
  // Source Caching ------------------------------------------------------------
  } else if(dispatcher_config.cache_policy() == 4) {
    if (cache_state.IsDatasetSourceCached(fingerprint)) {
      job_type = "GET_SOURCE";
    } else {
      job_type = "PUT_SOURCE";
    }
    return Status::OK();
  } else if (dispatcher_config.cache_policy() == 40) {
    job_type = "PUT_SOURCE";
    return Status::OK();
  } else if (dispatcher_config.cache_policy() == 41) {
    job_type = "GET_SOURCE";
    return
  Status::OK();
  }

  // ---------------------------------------------------------------------------
  // Cache policy = EASL (cache_policy==1)
  // ---------------------------------------------------------------------------

  // If dataset was previously cached, assume it was faster than compute
  // and decide to read.
  if (cache_state.IsDatasetCached(fingerprint)){
    job_type = "GET";
    return Status::OK();
  }

  // We always prefer source caching, so if we have full caching, it means
  // that was necessary. Thus, we check source caching after full caching.
  if (cache_state.IsDatasetSourceCached(fingerprint)) {
    job_type = "GET_SOURCE";
    return Status::OK();
  }
  std::shared_ptr<::tensorflow::data::easl::InputPipelineMetrics> job_metrics;
  Status s = metadata_store.GetInputPipelineMetricsByDatasetKey(dataset_key, 
    job_metrics);

  // We do not yet have the metrics for this dataset
  if(errors::IsNotFound(s)){
    job_type = "COMPUTE";
    return Status::OK();
  } else if (!s.ok()){
    return s;
  }

  // Compute metrics
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  std::shared_ptr<NodeMetrics> node_metrics;
  TF_RETURN_IF_ERROR(metadata_store.GetLastNodeMetricsByDatasetKey(
    dataset_key, node_metrics));

  size_t num_workers = (node_metrics->metrics_).size();
  DCHECK(num_workers > 0);

  uint64 compute_row_size = 0;
  uint64 compute_num_elements = 0;
  double compute_time_per_row_ms = 0;
  double compute_time_total_ms = 0;

  for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e : 
    node_metrics->metrics_){
    std::shared_ptr<NodeMetrics::Metrics> worker_metrics = e.second;
    // TODO average out row size here for datasets with varying row size?
    compute_row_size += worker_metrics->bytes_produced() / worker_metrics->num_elements();
    compute_time_per_row_ms += worker_metrics->active_time_ms();
    compute_num_elements += worker_metrics->num_elements();
  }

  compute_num_elements /= num_workers;
  compute_time_per_row_ms = compute_time_per_row_ms / num_workers;
  compute_time_total_ms = compute_time_per_row_ms * compute_num_elements;
  compute_row_size = compute_row_size / num_workers;

  VLOG(0) << "(Full caching) Row size " << compute_row_size;
  VLOG(0) << "Total compute time " << compute_time_total_ms;

  // Materialized Cache Read expecations
  double cache_read_time_per_row_ms = data::cache_model::GetTimePerRow(compute_row_size);
  double cache_read_time_total_ms = compute_num_elements * cache_read_time_per_row_ms;


  // IO metrics
  bool has_marker_node = false;
  bool is_gcs_limited = false;
  double source_cache_compute_time_ms = 0.0;
  std::shared_ptr<data::easl::InputPipelineMetrics> input_pipeline_metrics; 
  metadata_store.GetInputPipelineMetricsByDatasetKey(dataset_key, 
    input_pipeline_metrics);


  VLOG(0) << "(Full caching) Inferred GlusterFS read time "
          << cache_read_time_total_ms;
  
  if(input_pipeline_metrics->GetMarkerNodeName() != "") {
    VLOG(0) << "Found marker node name: " << input_pipeline_metrics->GetMarkerNodeName();
    has_marker_node = true;
    std::shared_ptr<data::easl::NodeMetrics> marker_node_metrics;
    metadata_store.GetMarkerNodeMetricsByDatasetKey(dataset_key, 
      marker_node_metrics);
  
    uint64 io_row_size = 0;
    uint64 io_num_elements = 0;
    double avg_io_bytes_per_s = 0.0; // Will not be used.
    double avg_io_time_total_ms = 0.0; // == avg_gcs_source_time_ms.
    for (auto& node_metrics : marker_node_metrics->metrics_) {
      avg_io_bytes_per_s += node_metrics.second->bytes_per_s();
      io_row_size += node_metrics.second->bytes_produced() /
        node_metrics.second->num_elements();
      avg_io_time_total_ms += node_metrics.second->active_time_ms();
      io_num_elements += node_metrics.second->num_elements();
    }
    avg_io_bytes_per_s /= num_workers;
    io_row_size /= num_workers;
    io_num_elements /= num_workers;
    avg_io_time_total_ms = (avg_io_time_total_ms * io_num_elements) / num_workers; // total io read time.
    double avg_io_bytes_per_active_time_ms = io_row_size / avg_io_time_total_ms;

    VLOG(0) << "Total GCS io time " << avg_io_time_total_ms;
    VLOG(0) << "GCS io throughput bytes/ms" << avg_io_bytes_per_s;
    VLOG(0) << "GCS io throughput per active time" << avg_io_bytes_per_active_time_ms;
    VLOG(0) << "GCS io row size " << io_row_size;

    if (compute_time_total_ms < avg_io_time_total_ms ||
        (compute_time_total_ms >= avg_io_time_total_ms && avg_io_bytes_per_active_time_ms > 524288 /*500MiB/sec*/)) {
        //avg_io_bytes_per_s < 0.001 * cache_model::GetGCSThrouhgput(0.95)) {
      // 1. compute active time is less than io time => pipeline is not io bound
      // 2. compute active time is more than io time, but io throughput is not at the limit. => bytes per active time is very high because always element in cache.
      // => compare directly with caching
      if (compute_time_total_ms < cache_read_time_total_ms) {
        job_type = "COMPUTE";
      } else {
        job_type = "PUT";
      }
      return Status::OK();
    } else {
      // pipeline is io bound, decide between source caching and caching
      double source_cache_io_time_per_row_ms = data::cache_model::GetTimePerRow(
          io_row_size);
      double source_cache_io_time_total_ms = source_cache_io_time_per_row_ms * io_num_elements;
      // This assumes the pipeline would be io bound with source caching
      // i.e. if true_compute_time_total > source_cache_io_time then source_cache_compute_time_total should be
      // true_compute_time_total.
      // => it is optimistic about source caching, need another metric: in_node_time without accounting for parallelism..
      double source_cache_compute_time_total_ms = source_cache_io_time_total_ms;
          //std::max(source_cache_io_time_per_row_ms, compute_time_per_row_ms);

      VLOG(0) << "GCS is limited";
      VLOG(0) << "IO row size " << io_row_size;
      VLOG(0) << "Estimated source cache io time: " << source_cache_io_time_total_ms;
      VLOG(0) << "Estimated source cache compute time: " << source_cache_compute_time_total_ms;

      if (source_cache_compute_time_total_ms < cache_read_time_total_ms) {
        job_type = "PUT_SOURCE";
      } else {
        job_type = "PUT";
      }
      return Status::OK();
    }

  } else {
    VLOG(0) << "No marker node found, choosing between put or compute";
    if (compute_time_per_row_ms < cache_read_time_per_row_ms) {
      job_type = "COMPUTE";
    } else {
      job_type = "PUT";
    }

    return Status::OK();
  }


  /**
  // We now make the caching decision: fastest option wins
  std::vector<double> v = {has_marker_node ? source_cache_compute_time_per_row_ms 
    : std::numeric_limits<double>::max(), cache_read_time_per_row_ms, 
    compute_time_per_row_ms};
  int minElementIndex = std::min_element(v.begin(), v.end()) - v.begin();

  switch(minElementIndex) {
    case 0: // This is the source cache
    job_type = "PUT_SOURCE";
    VLOG(0) << "Cache decision: SOURCE CACHING";
    break;
    case 1: // This is the full cache
    job_type = "PUT";
    VLOG(0) << "Cache decision: FULL CACHING";
    break;
    case 2: // This is the compute
    job_type = "COMPUTE";
    VLOG(0) << "Cache decision: COMPUTE";
    break;
    default:
    VLOG(0) << "Cache decision: In DEFAULT... Will throw error...";
    return errors::Unimplemented("Caching decision defaulted to last option... "
      "See DetermineJobType!");
  }

  return Status::OK();
   **/
}

Status DetermineElasticity(
  const std::string& job_type,
  const experimental::DispatcherConfig& dispatcher_config,
  const ::tensorflow::data::easl::MetadataStore& metadata_store,
  const std::string& dataset_key,
  const int64 available_workers,
  int64& worker_count) {
  using NodeMetrics = ::tensorflow::data::easl::NodeMetrics;
  using ModelMetrics = ::tensorflow::data::easl::ModelMetrics;

  // Give out max number of workers
  if(dispatcher_config.scaling_policy() == 2){
    worker_count = available_workers;
    return Status::OK();
  }

  // Check if we have any metrics for this dataset
  std::shared_ptr<data::easl::InputPipelineMetrics> job_metrics;
  Status s = metadata_store.GetInputPipelineMetricsByDatasetKey(
    dataset_key, job_metrics);

  // We do not yet have the metrics for this dataset --> use 1 worker
  if(errors::IsNotFound(s)) {
    VLOG(0) << "(DetermineElasticity) No metrics found for dataset " 
            << dataset_key << ". Will use 1 worker in " << job_type << " mode.";
    worker_count = 1;
    return Status::OK();
  } else if (!s.ok()) {
    VLOG(0) << "(DetermineElasticity) Another error has been thrown: " << s;
    return s;
  }

  // Pipeline stats: last TF node metrics
  VLOG(0) << "(DetermineElasticity) Before the GetLastTFNodeMetricsByDatasetKey";
  std::shared_ptr<NodeMetrics> last_tf_node_metrics;
  s = metadata_store.GetLastTFNodeMetricsByDatasetKey(
    dataset_key, last_tf_node_metrics);
  if (!s.ok()) {
    VLOG(0) << "(DetermineElasticity) Failed to get the last TF node metrics";
    return s;
  }
  VLOG(0) << "(DetermineElasticity) After the GetLastTFNodeMetricsByDatasetKey " << last_tf_node_metrics;
  size_t num_workers = (last_tf_node_metrics->metrics_).size();
  VLOG(0) << "(DetermineElasticity) After the GetLastTFNodeMetricsByDatasetKey size get " << num_workers;
  DCHECK(num_workers > 0);
  VLOG(0) << "(DetermineElasticity) After the GetLastTFNodeMetricsByDatasetKey size check";

  // Model stats
  double client_throughput = 0.0;
  std::shared_ptr<ModelMetrics> model_metrics;
  TF_RETURN_IF_ERROR(
    metadata_store.GetModelMetricsByDatasetKey(
      dataset_key, model_metrics));
  VLOG(0) << "(DetermineElasticity) After the GetModelMetricsByDatasetKey";

  // Get the client throughput 
  for(std::pair<int64, std::shared_ptr<ModelMetrics::Metrics>> e : 
    model_metrics->metrics_) {
    std::shared_ptr<ModelMetrics::Metrics> client_metrics = e.second;
    client_throughput += 1000.0 / client_metrics->inter_arrival_time_ms();
    VLOG(0) << "(DetermineElasticity) Client inter arrival time " 
            << client_metrics->inter_arrival_time_ms();
  }
  VLOG(0) << "(DetermineElasticity) Total client throughput demand " 
          << client_throughput;

  if (job_type == "COMPUTE" || job_type == "PUT") {
    VLOG(0) << "(DetermineElasticity) In COMPUTE or PUT branch with case " 
            << job_type;
    double avg_worker_throughput = 0.0;

    // Get the average throughput for a worker
    for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e : 
      last_tf_node_metrics->metrics_) {
      std::shared_ptr<NodeMetrics::Metrics> worker_metrics = e.second;
      avg_worker_throughput += 1000.0 / worker_metrics->in_prefix_time_ms();
    }
    avg_worker_throughput /= num_workers;
    VLOG(0) << "(DetermineElasticity) Average worker throughput "
            << avg_worker_throughput;

    // Infer the number of workers required to sustain the model
    worker_count = std::max<int64>(ceil(client_throughput / avg_worker_throughput - 
      worker_count_alpha_), 1);
  } else {
    VLOG(0) << "(DetermineElasticity) In GET branch."; 

    // Get last user node metrics
    std::shared_ptr<NodeMetrics> last_node_metrics;
    TF_RETURN_IF_ERROR(
      metadata_store.GetLastNodeMetricsByDatasetKey(
        dataset_key, last_node_metrics));
    DCHECK(num_workers == (last_node_metrics->metrics_).size());

    // Get the time per row and the TF nodes overhead
    uint64 row_size = 0;
    double tf_nodes_overhead_ms = 0.0;

    for(std::pair<std::string, std::shared_ptr<NodeMetrics::Metrics>> e : 
      last_node_metrics->metrics_) {
      std::shared_ptr<NodeMetrics::Metrics> worker_metrics = e.second;
      std::shared_ptr<NodeMetrics::Metrics> worker_metrics_tf_node = 
        last_tf_node_metrics->metrics_[e.first];

      row_size += worker_metrics->bytes_produced() / 
        worker_metrics->num_elements();
      tf_nodes_overhead_ms += worker_metrics_tf_node->in_prefix_time_ms() 
        - worker_metrics->in_prefix_time_ms();
    }

    row_size /= num_workers;
    tf_nodes_overhead_ms /= num_workers;
    double cache_read_time_per_row_ms = data::cache_model::GetTimePerRow(
      row_size);
    
    VLOG(0) << "(DetermineElasticity) Average row size " << row_size;
    VLOG(0) << "(DetermineElasticity) Average read time per item " 
            << cache_read_time_per_row_ms;
    VLOG(0) << "(DetermineElasticity) Average TF overhead " 
            << tf_nodes_overhead_ms;

    // Infer the worker count for the cache GET use case
    worker_count = std::max<int64>(ceil(client_throughput * 
      (cache_read_time_per_row_ms + tf_nodes_overhead_ms) 
      / 1000.0 - worker_count_alpha_), 1);
  }
  VLOG(0) << "(DetermineElasticity) Inferred workers " << worker_count;
  
  return Status::OK();
}

Status AddPutOperator(const DatasetDef& dataset,
                      const uint64 fingerprint,
                      const experimental::DispatcherConfig& dispatcher_config,
                      DatasetDef& updated_dataset) {
  // TODO remove this.
  //updated_dataset = dataset;
  //return Status::OK();
  VLOG(1) << "(AddPutOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer
  tensorflow::grappler::easl::AddPutOp optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(
      absl::StrCat(dispatcher_config.cache_path(), "/materialized/", fingerprint));
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  //DoBFS(sink, *graph_def, "AddPutOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  //DoBFS(sink, *graph_def, "AfterAddPutOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(1) << "(AddPutOperator) At the end of the method";

  return Status::OK();
}


Status AddGetOperator(const DatasetDef& dataset,
                      const uint64 fingerprint,
                      const experimental::DispatcherConfig& dispatcher_config,
                      DatasetDef& updated_dataset){
  // TODO remove this.
  //updated_dataset = dataset;
  //return Status::OK();

  VLOG(1) << "(AddGetOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer  
  tensorflow::grappler::easl::AddGetOp optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(
      absl::StrCat(dispatcher_config.cache_path(), "/materialized/", fingerprint));
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  //DoBFS(sink, *graph_def, "AddGetOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  //DoBFS(sink, *graph_def, "AfterAddGetOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(1) << "(AddGetOperator) At the end of the method";

  return Status::OK();
}


Status AddPutOperatorAtMarker(const DatasetDef& dataset,
                              const uint64 fingerprint,
                              const std::string& marker_type,
                              const experimental::DispatcherConfig& dispatcher_config,
                              DatasetDef& updated_dataset) {
  VLOG(0) << "(AddPutOperatorAtMarker) At the beginning of the method";

  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer
  tensorflow::grappler::easl::AddPutOpAtMarker optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // Set correct path
  std::string cache_path;
  if(marker_type == "source_cache"){
    cache_path = absl::StrCat(dispatcher_config.cache_path(), "/source/", fingerprint);
  } else {
    absl::StrCat(dispatcher_config.cache_path(), "/materialized/", fingerprint);
  }

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(cache_path);
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());
  (*(config.mutable_parameter_map()))["marker_type"].set_placeholder(
      marker_type);

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS for debugging
  //DoBFS(sink, *graph_def, "AddPutAtMarkerOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  //DoBFS(sink, *graph_def, "AfterAddPutAtMarkerOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  VLOG(0) << "(AddPutOperatorAtMarker) At the end of the method";

  return Status::OK();
}


Status AddGetOperatorAtMarker(
    const DatasetDef& dataset,
    const uint64 fingerprint,
    const std::string& marker_type,
    const experimental::DispatcherConfig& dispatcher_config,
    DatasetDef& updated_dataset){
  // TODO remove this.
  //updated_dataset = dataset;
  //return Status::OK();

  VLOG(0) << "(AddGetOperator) At the start of the method";
  // Copy over the original dataset
  updated_dataset = dataset;

  // Initialize the optimizer
  tensorflow::grappler::easl::AddGetOpAtMarker optimizer;
  // Transfer arguments from dispatcher config to optimizer config.
  tensorflow::RewriterConfig_CustomGraphOptimizer config;

  // Set correct path
  std::string cache_path;
  if(marker_type == "source_cache"){
    cache_path = absl::StrCat(dispatcher_config.cache_path(), "/source/", fingerprint);
  } else {
    absl::StrCat(dispatcher_config.cache_path(), "/materialized/", fingerprint);
  }

  // TODO - set path where to store graph.
  (*(config.mutable_parameter_map()))["path"].set_placeholder(cache_path);
  (*(config.mutable_parameter_map()))["cache_format"].set_i(
      dispatcher_config.cache_format());
  (*(config.mutable_parameter_map()))["cache_compression"].set_i(
      dispatcher_config.cache_compression());
  (*(config.mutable_parameter_map()))["cache_ops_parallelism"].set_i(
      dispatcher_config.cache_ops_parallelism());
  (*(config.mutable_parameter_map()))["marker_type"].set_placeholder(
      marker_type);

  optimizer.Init(&config);

  // Get the graph def and wrap it in a GrapplerItem
  GraphDef* graph_def = updated_dataset.mutable_graph();
  std::string output_node;

  // Find the output node; the one before '_Retval'
  for (const auto& node : graph_def->node()) {
    if (node.op() == "_Retval") {
      output_node = node.input(0);
    }
  }

  // Create a 'Sink' node and attatch it to the real output
  NodeDef* sink = graph_def->mutable_node()->Add();
  tensorflow::grappler::graph_utils::SetUniqueGraphNodeName("Sink", graph_def,
                                                            sink);
  sink->set_op("Identity");
  sink->add_input(output_node);
  (*sink->mutable_attr())["T"].set_type(DT_VARIANT);

  // Do BFS
  //DoBFS(sink, *graph_def, "AddGetOperator");

  // Create the MuttableGraphView
  tensorflow::grappler::MutableGraphView graph(graph_def);
  optimizer.ApplyOptimization(graph, sink, graph_def);

  // Do BFS
  //DoBFS(sink, *graph_def, "AfterAddGetOperator");

  // Disconnect the 'Sink' node
  // sink->mutable_input()->Clear();
  // VLOG(0) << "(AddGetOperatorAtMarker) At the end of the method";

  return Status::OK();
}



} // namespace cache_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow
