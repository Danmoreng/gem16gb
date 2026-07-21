#include "test.h"

#include <string>

#include "util/json.h"

void RunJsonTests() {
  auto parsed = g4::json::Parse(R"({"name":"g4","shape":[2,16],"enabled":true,"unicode":"\uD83D\uDE80"})");
  G4_CHECK(parsed.ok());
  if (parsed.ok()) {
    G4_CHECK(parsed.value().is_object());
    G4_CHECK(parsed.value().find("name") != nullptr);
    G4_CHECK(parsed.value().find("name")->as_string() == "g4");
    G4_CHECK(parsed.value().find("shape")->as_array()[1].as_integer() == 16);
    G4_CHECK(parsed.value().find("unicode")->as_string() == "\xF0\x9F\x9A\x80");
  }

  G4_CHECK(!g4::json::Parse(R"({"duplicate":1,"duplicate":2})").ok());
  G4_CHECK(!g4::json::Parse(R"("\uD800")").ok());
  G4_CHECK(!g4::json::Parse("[01]").ok());
  G4_CHECK(!g4::json::Parse("{} trailing").ok());
  G4_CHECK(!g4::json::Parse(std::string{"\"\xC0\x80\"", 4}).ok());
  G4_CHECK(g4::json::Escape("a\n\"b") == "a\\n\\\"b");
}
