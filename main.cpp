#include "similarity_transform.hpp"
#include "utils.hpp"
#include <iostream>

using namespace sycl;

const uint N = 4;
const uint B = 4;

int main() {
  device d{default_selector{}};
  queue q{d};
  std::cout << "running on " << d.get_info<info::device::name>() << std::endl;

  return 0;
}
