#include <iostream>

#include "test.h"

void RunJsonTests();
void RunSafetensorsTests();

int main() {
  RunJsonTests();
  RunSafetensorsTests();
  if (g4::test::failures != 0) {
    std::cerr << g4::test::failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "all host unit tests passed\n";
  return 0;
}

