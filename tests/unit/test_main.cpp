#include <iostream>

#include "test.h"

void RunJsonTests();
void RunSafetensorsTests();

int main() {
  RunJsonTests();
  RunSafetensorsTests();
  if (gem16gb::test::failures != 0) {
    std::cerr << gem16gb::test::failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all host unit tests passed\n";
  return 0;
}

