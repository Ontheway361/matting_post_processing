# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/lujie/anaconda3/envs/lujie/bin/cmake

# The command to remove a file.
RM = /home/lujie/anaconda3/envs/lujie/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/build

# Include any dependencies generated for this target.
include CMakeFiles/sky_matting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sky_matting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sky_matting.dir/flags.make

CMakeFiles/sky_matting.dir/main.cpp.o: CMakeFiles/sky_matting.dir/flags.make
CMakeFiles/sky_matting.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sky_matting.dir/main.cpp.o"
	/home/lujie/anaconda3/envs/lujie/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sky_matting.dir/main.cpp.o -c /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/main.cpp

CMakeFiles/sky_matting.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sky_matting.dir/main.cpp.i"
	/home/lujie/anaconda3/envs/lujie/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/main.cpp > CMakeFiles/sky_matting.dir/main.cpp.i

CMakeFiles/sky_matting.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sky_matting.dir/main.cpp.s"
	/home/lujie/anaconda3/envs/lujie/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/main.cpp -o CMakeFiles/sky_matting.dir/main.cpp.s

CMakeFiles/sky_matting.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/sky_matting.dir/main.cpp.o.requires

CMakeFiles/sky_matting.dir/main.cpp.o.provides: CMakeFiles/sky_matting.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/sky_matting.dir/build.make CMakeFiles/sky_matting.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/sky_matting.dir/main.cpp.o.provides

CMakeFiles/sky_matting.dir/main.cpp.o.provides.build: CMakeFiles/sky_matting.dir/main.cpp.o


CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o: CMakeFiles/sky_matting.dir/flags.make
CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o: ../global-matting/globalmatting.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o"
	/home/lujie/anaconda3/envs/lujie/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o -c /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/global-matting/globalmatting.cpp

CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.i"
	/home/lujie/anaconda3/envs/lujie/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/global-matting/globalmatting.cpp > CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.i

CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.s"
	/home/lujie/anaconda3/envs/lujie/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/global-matting/globalmatting.cpp -o CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.s

CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o.requires:

.PHONY : CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o.requires

CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o.provides: CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o.requires
	$(MAKE) -f CMakeFiles/sky_matting.dir/build.make CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o.provides.build
.PHONY : CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o.provides

CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o.provides.build: CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o


# Object files for target sky_matting
sky_matting_OBJECTS = \
"CMakeFiles/sky_matting.dir/main.cpp.o" \
"CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o"

# External object files for target sky_matting
sky_matting_EXTERNAL_OBJECTS =

sky_matting: CMakeFiles/sky_matting.dir/main.cpp.o
sky_matting: CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o
sky_matting: CMakeFiles/sky_matting.dir/build.make
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_xphoto.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_xobjdetect.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_tracking.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_surface_matching.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_structured_light.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_stereo.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_saliency.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_rgbd.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_reg.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_plot.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_optflow.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_line_descriptor.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_hdf.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_fuzzy.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_dpm.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_dnn.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_datasets.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_ccalib.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_bioinspired.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_bgsegm.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_aruco.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_videostab.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_superres.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_stitching.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_photo.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_text.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_face.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_ximgproc.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_xfeatures2d.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_shape.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_video.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_objdetect.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_calib3d.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_features2d.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_ml.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_highgui.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_videoio.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_imgcodecs.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_imgproc.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_flann.so.3.1.0
sky_matting: /home/lujie/anaconda3/envs/lujie/lib/libopencv_core.so.3.1.0
sky_matting: CMakeFiles/sky_matting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable sky_matting"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sky_matting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sky_matting.dir/build: sky_matting

.PHONY : CMakeFiles/sky_matting.dir/build

CMakeFiles/sky_matting.dir/requires: CMakeFiles/sky_matting.dir/main.cpp.o.requires
CMakeFiles/sky_matting.dir/requires: CMakeFiles/sky_matting.dir/global-matting/globalmatting.cpp.o.requires

.PHONY : CMakeFiles/sky_matting.dir/requires

CMakeFiles/sky_matting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sky_matting.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sky_matting.dir/clean

CMakeFiles/sky_matting.dir/depend:
	cd /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0 /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0 /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/build /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/build /home/lujie/project/matting_Algorithm/release_matting/global_matting_release/test_algorithm/matting_version_2.0/build/CMakeFiles/sky_matting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sky_matting.dir/depend

