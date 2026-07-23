#include <iostream>

#include "test.h"

void RunJsonTests();
void RunMemoryPlanTests();
void RunNvfp4Tests();
void RunSafetensorsTests();
void RunSm120LayoutTests();

int main() {
  RunJsonTests();
  RunMemoryPlanTests();
  RunNvfp4Tests();
  RunSafetensorsTests();
  RunSm120LayoutTests();
  if (gem16gb::test::failures != 0) {
    std::cerr << gem16gb::test::failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all host unit tests passed\n";
  return 0;
}

