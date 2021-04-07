//
// Created by simon on 07.04.21.
//

#ifndef ML_INPUT_DATA_SERVICE_ARROW_UTIL_H
#define ML_INPUT_DATA_SERVICE_ARROW_UTIL_H


// used macros ---------------
#define CHECK_ARROW(arrow_status)             \
  do {                                        \
    arrow::Status _s = (arrow_status);        \
    if (!_s.ok()) {                           \
      return errors::Internal(_s.ToString()); \
    }                                         \
  } while (false)

#define ARROW_ASSIGN_CHECKED(lhs, rexpr)                                              \
  ARROW_ASSIGN_CHECKED_IMPL(ARROW_ASSIGN_OR_RAISE_NAME(_error_or_value, __COUNTER__), \
                             lhs, rexpr);

#define ARROW_ASSIGN_CHECKED_IMPL(result_name, lhs, rexpr) \
  auto&& result_name = (rexpr);                             \
  CHECK_ARROW((result_name).status());              \
  lhs = std::move(result_name).ValueUnsafe();




#endif //ML_INPUT_DATA_SERVICE_ARROW_UTIL_H
