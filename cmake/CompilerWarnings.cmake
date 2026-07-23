function(gem16gb_set_project_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${GEM16GB_WARNINGS_AS_ERRORS}>>:/WX>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${GEM16GB_WARNINGS_AS_ERRORS}>>:-Xcompiler=/WX>
    )
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow;-Wformat=2;-Wundef>
      $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${GEM16GB_WARNINGS_AS_ERRORS}>>:-Werror>
    )
  endif()
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endfunction()

