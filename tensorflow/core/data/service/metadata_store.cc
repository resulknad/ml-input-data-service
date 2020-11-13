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
#include "tensorflow/core/data/service/metadata_store.h"

#include <memory>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {

MetadataStore::MetadataStore() {}

Status MetadataStore::UpdateMetadata(const uint64& fingerprint, const int64& update){
  metadata_by_fingerprint_[fingerprint] = update;
  return Status::OK();
}

Status MetadataStore::MetadataFromFingerprint(
    uint64 fingerprint,
    std::shared_ptr<const int64>& metadata) const {

  auto it = metadata_by_fingerprint_.find(fingerprint);
  if(it == metadata_by_fingerprint_.end()){
    return errors::NotFound("Dataset fingerprint ", fingerprint, " not found")
  }

  *metadata = it->second;
  return Status::OK();
}


}  // namespace data
}  // namespace tensorflow
