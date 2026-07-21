function(g4_set_project_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 $<$<BOOL:${G4_WARNINGS_AS_ERRORS}>:/WX>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow;-Wformat=2;-Wundef>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${G4_WARNINGS_AS_ERRORS}>>:-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
  endif()
endfunction()

