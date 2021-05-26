//
// Created by damien-aymon on 26.05.21.
//

#ifndef TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_MODEL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_MODEL_H_

#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/lib/core/status.h"


namespace tensorflow {
namespace data {
namespace cache_model {

double GetTimePerRow(uint64 row_size);

} // cache_model
} // data
} // tensorflow


#endif // TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_MODEL_H_
