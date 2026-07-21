#include "test.h"

#include <string>

#include "util/json.h"

void RunJsonTests() {
  auto parsed = gem16gb::json::Parse(R"({"name":"gem16gb","shape":[2,16],"enabled":true,"unicode":"\uD83D\uDE80"})");
  GEM16GB_CHECK(parsed.ok());
  if (parsed.ok()) {
    GEM16GB_CHECK(parsed.value().is_object());
    GEM16GB_CHECK(parsed.value().find("name") != nullptr);
    GEM16GB_CHECK(parsed.value().find("name")->as_string() == "gem16gb");
    GEM16GB_CHECK(parsed.value().find("shape")->as_array()[1].as_integer() == 16);
    GEM16GB_CHECK(parsed.value().find("unicode")->as_string() == "\xF0\x9F\x9A\x80");
  }

  GEM16GB_CHECK(!gem16gb::json::Parse(R"({"duplicate":1,"duplicate":2})").ok());
  GEM16GB_CHECK(!gem16gb::json::Parse(R"("\uD800")").ok());
  GEM16GB_CHECK(!gem16gb::json::Parse("[01]").ok());
  GEM16GB_CHECK(!gem16gb::json::Parse("{} trailing").ok());
  GEM16GB_CHECK(!gem16gb::json::Parse(std::string{"\"\xC0\x80\"", 4}).ok());
  GEM16GB_CHECK(gem16gb::json::Escape("a\n\"b") == "a\\n\\\"b");
}
