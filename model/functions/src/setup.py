import os
import glob
import torch
import shutil
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = os.path.dirname(os.path.abspath(__file__))
# setup.py is already in src directory, so use this_dir directly
extensions_dir = this_dir

# Main binding file
main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
# CPU source files in subdirectories
source_cpu = glob.glob(os.path.join(extensions_dir, "*", "*.cpp"))
# CUDA source files in subdirectories
source_cuda = glob.glob(os.path.join(extensions_dir, "*", "*.cu"))

if torch.cuda.is_available():
    if 'LD_LIBRARY_PATH' not in os.environ:
        raise Exception('LD_LIBRARY_PATH is not set.')
    cuda_lib_path = os.environ['LD_LIBRARY_PATH'].split(':')
    sources = source_cpu + source_cuda + main_file

    # Remove duplicates and filter empty
    sources = list(set([s for s in sources if s]))

    if len(sources) == 0:
        raise FileNotFoundError(
            f"No source files found in '{extensions_dir}'.\n"
            f"Main files: {main_file}\n"
            f"CPU sources: {source_cpu}\n"
            f"CUDA sources: {source_cuda}\n"
            "Please check that all source files are present."
        )
else:
    raise Exception('This implementation is only avaliable for CUDA devices.')

print(f"Found {len(sources)} source files:")
for s in sorted(sources):
    print(f"  - {s}")


# Custom BuildExtension to skip CUDA version check
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # Temporarily disable CUDA version check
        import torch.utils.cpp_extension as cpp_extension
        original_check = cpp_extension._check_cuda_version

        # Monkey patch to skip version check
        def skip_check(*args, **kwargs):
            pass

        cpp_extension._check_cuda_version = skip_check

        try:
            # Call parent build_extensions
            super().build_extensions()
        finally:
            # Restore original check function
            cpp_extension._check_cuda_version = original_check


setup(
    name='tree_filter',
    version="0.1",
    description="learnable tree filter for pytorch",
    ext_modules=[
        CUDAExtension(
            name='tree_filter_cuda',
            include_dirs=[
                extensions_dir,
                os.path.join(extensions_dir, 'mst'),
                os.path.join(extensions_dir, 'bfs'),
                os.path.join(extensions_dir, 'refine')
            ],
            sources=sources,
            library_dirs=cuda_lib_path,
            extra_compile_args={
                # PyTorch 2.x requires C++17
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-std=c++17', '--expt-relaxed-constexpr', '--expt-extended-lambda']
            })
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    })
