//
// Created by aymond on 26.11.21.
//

#ifndef ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_DATA_SERVICE_EASL_SCALING_UTILS_H_
#define ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_DATA_SERVICE_EASL_SCALING_UTILS_H_


#include <string>
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/data/service/easl/dispatcher_cache_state.h"
#include "tensorflow/core/data/service/easl/metadata_store.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/protobuf/service_config.pb.h"


namespace tensorflow {
namespace data {
namespace service {
namespace easl{
namespace scaling_utils {

Status DetermineElasticity(
    const std::string& job_type,
    const experimental::DispatcherConfig& dispatcher_config,
    const ::tensorflow::data::easl::MetadataStore& metadata_store,
    const std::string& dataset_key,
    const int64 available_workers,
    int64& worker_count);

Status DynamicWorkerCountUpdate(
    const std::string& job_type,
    const experimental::DispatcherConfig& dispatcher_config,
    const ::tensorflow::data::easl::MetadataStore& metadata_store,
    const int64 current_worker_count,
    int64& worker_count
    );


} // namespace scaling_utils
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_TENSORFLOW_CORE_DATA_SERVICE_EASL_SCALING_UTILS_H_
