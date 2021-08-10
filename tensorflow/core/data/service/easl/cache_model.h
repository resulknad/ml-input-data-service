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

// Returns the GCS throughput (egress) in bytes per second; alpha is fraction 
// \in [0, 1] indicating how much of the GCS throughput limit is being used  
double GetGCSThrouhgput(double alpha);

} // cache_model
} // data
} // tensorflow


#endif // TENSORFLOW_CORE_DATA_SERVICE_EASL_CACHE_MODEL_H_
