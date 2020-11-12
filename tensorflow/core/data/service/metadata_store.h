/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_service.h"
#include "tensorflow/core/data/service/journal.h"
#include "tensorflow/core/data/service/journal.pb.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// A class encapsulating the metadata store of the tf.data service.
// TODO(damien-aymon)
// - persist this store to disk for fault recovery.


class MetadataStore {
 public:
  MetadataStore();
  MetadataStore(const MetadataStore&) = delete;
  MetadataStore& operator=(const MetadataStore&) = delete;

  // Sets the metadata for this fingerprint.
  Status UpdateMetadata(const uint64& fingerprint, const int64& update);
  // Returns the (dummy for now) metadata by fingerprint.
  Status MetadataFromFingerprint(uint64 fingerprint,
                                 std::shared_ptr<const int64>& metadata) const;
  
  private:
  // Some dummy metadata.
  absl::flat_hash_map<uint64, int64> metadata_by_fingerprint_;

};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DISPATCHER_STATE_H_
