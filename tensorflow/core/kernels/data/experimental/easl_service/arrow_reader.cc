//
// Created by simon on 30.03.21.
//

#include "tensorflow/core/profiler/lib/traceme.h"
#include "arrow_reader.h"
#include "arrow/api.h"

namespace tensorflow {
namespace data {
namespace easl{
namespace service_cache_util {

void ArrowReader::PrintTestLog() {
    VLOG(0) << "ARROW - TestLog\nArrow Version: " << arrow::GetBuildInfo().version_string;
}


} // namespace service_cache_util
} // namespace easl
} // namespace data
} // namespace tensorflow
