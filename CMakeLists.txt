# SPDX-FileCopyrightText: Copyright (c) 2026 zozo
# SPDX-License-Identifier: Apache-2.0

# Builds the modules of this collection together, from the repository root. Another
# project reaches it with FetchContent (by hand, or through the Holoscan CLI's
# holohub_declare_external_module()) or with add_subdirectory(); it also builds on its own.
#
# Every modules/<name>/ is added when it is enabled. Its switches come from its
# metadata.json, so a new module needs no change here:
#   MODULE_holoscan_<name>  the whole module ("-" in <name> becomes "_"), as the CLI names it
#   OP_<operator>           one per entry of module.operator_names, as HoloHub names them;
#                           any of them ON enables its module
# Built on its own, every module is enabled (BUILD_ALL, ON); included by another project,
# only those it enables, unless its BUILD_ALL is ON. A module enabled here is built as part
# of this build, and its targets are used as holoscan::<target>, without find_package().
# What a module must provide for this is in modules/README.md.

cmake_minimum_required(VERSION 4.0)

# A consumer that declares two modules of this repository as two dependencies reaches
# this file twice. The first copy already added every module the build enables, from
# the OP_ and MODULE_ options set by then; a second would redefine their targets, so it
# adds nothing. Both declarations must name the same commit.
get_property(_holoscan_zozo_modules_root GLOBAL PROPERTY HOLOSCAN_ZOZO_MODULES_ROOT)
if(_holoscan_zozo_modules_root)
  message(
    STATUS
    "holoscan-zozo-modules: already added from ${_holoscan_zozo_modules_root};"
    " the copy in ${CMAKE_CURRENT_SOURCE_DIR} adds nothing"
  )
  return()
endif()
set_property(GLOBAL PROPERTY HOLOSCAN_ZOZO_MODULES_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

project(holoscan_zozo_modules LANGUAGES NONE)

if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_ALL "Build every module, whatever its MODULE_ and OP_ options say" ON)
  # A module's tests register in its own folder; this makes `ctest` here run them.
  enable_testing()
endif()

# Adds modules/<folder>/ when it is enabled. A function, so that what one module sets
# stays out of the next one and out of the including project.
function(holoscan_zozo_modules_add folder)
  set(dir "${CMAKE_CURRENT_SOURCE_DIR}/modules/${folder}")
  if(NOT EXISTS "${dir}/metadata.json" OR NOT EXISTS "${dir}/CMakeLists.txt")
    message(FATAL_ERROR "${dir}: a module needs metadata.json and CMakeLists.txt")
  endif()
  file(READ "${dir}/metadata.json" metadata)
  string(JSON name ERROR_VARIABLE error GET "${metadata}" module name)
  if(error OR NOT name STREQUAL "holoscan-${folder}")
    message(FATAL_ERROR "${dir}/metadata.json: module.name must be holoscan-${folder}")
  endif()

  set(operators "")
  string(JSON type ERROR_VARIABLE error TYPE "${metadata}" module operator_names)
  if(NOT error)
    if(NOT type STREQUAL "ARRAY")
      message(FATAL_ERROR "${dir}/metadata.json: module.operator_names must be an array")
    endif()
    string(JSON count LENGTH "${metadata}" module operator_names)
    if(count GREATER 0)
      math(EXPR last "${count} - 1")
      foreach(index RANGE ${last})
        string(JSON operator GET "${metadata}" module operator_names ${index})
        string(MAKE_C_IDENTIFIER "${operator}" identifier)
        if(NOT operator STREQUAL identifier)
          message(
            FATAL_ERROR
            "${dir}/metadata.json: operator name '${operator}' is not a C identifier,"
            " so it cannot name an OP_ option"
          )
        endif()
        get_property(owner GLOBAL PROPERTY "HOLOSCAN_ZOZO_MODULES_OP_${operator}")
        if(owner)
          message(FATAL_ERROR "Operator ${operator} is named by both ${owner} and ${name}")
        endif()
        set_property(GLOBAL PROPERTY "HOLOSCAN_ZOZO_MODULES_OP_${operator}" "${name}")
        list(APPEND operators "${operator}")
      endforeach()
    endif()
  endif()

  # holoscan-point-cloud gives MODULE_holoscan_point_cloud and holoscan::point_cloud.
  string(MAKE_C_IDENTIFIER "${name}" slug)
  string(MAKE_C_IDENTIFIER "${folder}" target)
  set(default OFF)
  if(BUILD_ALL)
    set(default ON)
  endif()
  option(MODULE_${slug} "Build the ${name} module, all of its operators" ${default})
  set(requested "")
  foreach(operator IN LISTS operators)
    option(OP_${operator} "Build the ${name} module for its ${operator}" OFF)
    if(MODULE_${slug} OR OP_${operator})
      list(APPEND requested "${operator}")
    endif()
  endforeach()
  if(NOT MODULE_${slug} AND NOT requested)
    message(STATUS "holoscan-zozo-modules: ${name} not enabled")
    return()
  endif()
  # Already in this build: the including project added the module's own folder, as the
  # Holoscan CLI does for HOLOSCAN_CLI_LOCAL_HOLOSCAN_<NAME>=<checkout>/modules/<name>.
  if(TARGET holoscan::${target})
    message(STATUS "holoscan-zozo-modules: ${name} is already in this build; not adding it")
    return()
  endif()

  # The operators asked for, which a module may read to leave out an optional part.
  set(HOLOSCAN_ZOZO_MODULES_OPERATORS "${requested}")
  message(STATUS "holoscan-zozo-modules: adding ${name} (operators: ${requested})")
  add_subdirectory("${dir}")
  if(NOT TARGET holoscan::${target})
    message(
      FATAL_ERROR
      "${dir}: the module must define holoscan::${target} in the build tree"
      " (add_library(holoscan::${target} ALIAS ...)); see modules/README.md"
    )
  endif()
  set_property(GLOBAL APPEND PROPERTY HOLOSCAN_ZOZO_MODULES_ADDED "${name}")
endfunction()

# Adds modules/<folder>/ after the modules it depends on: each entry of its
# module.dependencies without a `source` names one, by folder, as CI reads them. A name
# that is not a module here, or a cycle, fails the configure.
function(holoscan_zozo_modules_visit folder)
  get_property(state GLOBAL PROPERTY "HOLOSCAN_ZOZO_MODULES_VISIT_${folder}")
  if(state STREQUAL "done")
    return()
  endif()
  get_property(path GLOBAL PROPERTY HOLOSCAN_ZOZO_MODULES_VISITING)
  if(state STREQUAL "visiting")
    list(APPEND path "${folder}")
    list(JOIN path " -> " cycle)
    message(FATAL_ERROR "Modules depend on each other in a cycle: ${cycle}")
  endif()
  set_property(GLOBAL PROPERTY "HOLOSCAN_ZOZO_MODULES_VISIT_${folder}" visiting)
  set_property(GLOBAL APPEND PROPERTY HOLOSCAN_ZOZO_MODULES_VISITING "${folder}")

  set(metadata_file "${CMAKE_CURRENT_SOURCE_DIR}/modules/${folder}/metadata.json")
  if(EXISTS "${metadata_file}")
    file(READ "${metadata_file}" metadata)
    string(JSON count ERROR_VARIABLE error LENGTH "${metadata}" module dependencies)
    if(NOT error AND count GREATER 0)
      math(EXPR last "${count} - 1")
      foreach(index RANGE ${last})
        string(
          JSON source
          ERROR_VARIABLE no_source
          TYPE "${metadata}"
          module
          dependencies
          ${index}
          source
        )
        if(NOT no_source)
          continue() # Has a source: from another repository, not ordered here.
        endif()
        string(JSON dependency GET "${metadata}" module dependencies ${index} name)
        if(
          dependency MATCHES "^\\."
          OR NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/modules/${dependency}"
        )
          message(
            FATAL_ERROR
            "${metadata_file}: module.dependencies names '${dependency}', which is not a"
            " module in modules/ and has no source"
          )
        endif()
        holoscan_zozo_modules_visit("${dependency}")
      endforeach()
    endif()
  endif()

  holoscan_zozo_modules_add("${folder}")
  get_property(path GLOBAL PROPERTY HOLOSCAN_ZOZO_MODULES_VISITING)
  list(POP_BACK path)
  set_property(GLOBAL PROPERTY HOLOSCAN_ZOZO_MODULES_VISITING "${path}")
  set_property(GLOBAL PROPERTY "HOLOSCAN_ZOZO_MODULES_VISIT_${folder}" done)
endfunction()

# Every folder in modules/ but a hidden one is a module, as CI sees them. Each is added
# after the modules it depends on, so that it may use their targets; otherwise in name
# order.
file(
  GLOB _holoscan_zozo_modules_folders
  LIST_DIRECTORIES true
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/modules"
  "${CMAKE_CURRENT_SOURCE_DIR}/modules/*"
)
list(SORT _holoscan_zozo_modules_folders)
foreach(_folder IN LISTS _holoscan_zozo_modules_folders)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/modules/${_folder}" AND NOT _folder MATCHES "^\\.")
    holoscan_zozo_modules_visit("${_folder}")
  endif()
endforeach()

get_property(_holoscan_zozo_modules_added GLOBAL PROPERTY HOLOSCAN_ZOZO_MODULES_ADDED)
if(NOT _holoscan_zozo_modules_added)
  message(STATUS "holoscan-zozo-modules: no module enabled")
endif()
