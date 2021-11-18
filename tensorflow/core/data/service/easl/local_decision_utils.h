//
// Created by Muyu Li on 16.11.21.
//

#ifndef ML_INPUT_DATA_SERVICE_LOCAL_DECISION_UTILS_H
#define ML_INPUT_DATA_SERVICE_LOCAL_DECISION_UTILS_H

#include <string>
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/data/service/easl/metadata_store.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/dispatcher_state.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {
namespace service {
namespace easl {
namespace local_decision {

Status DecideIfLocal(
        const experimental::DispatcherConfig& dispatcher_config,
        const ::tensorflow::data::easl::MetadataStore& metadata_store,
        const std::string& dataset_key,
        bool& if_local);

} // namespace local_decision
} // namespace easl
} // namespace service
} // namespace data
} // namespace tensorflow

#endif //ML_INPUT_DATA_SERVICE_LOCAL_DECISION_UTILS_H
