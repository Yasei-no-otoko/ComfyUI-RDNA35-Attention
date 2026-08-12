$pythonSitePackages = "C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\Lib\site-packages"
$rocmCore = "$pythonSitePackages\_rocm_sdk_core"
$rocmDevel = "$pythonSitePackages\_rocm_sdk_devel"
$rocmLibraries = "$pythonSitePackages\_rocm_sdk_libraries_gfx1151"

$env:PATH = "$rocmLibraries\bin;$rocmCore\bin;$rocmDevel\bin;$env:PATH"
$env:INCLUDE = "$rocmCore\include;$rocmDevel\include;$env:INCLUDE"
$env:LIB = "$rocmCore\lib;$rocmDevel\lib;$rocmLibraries\lib;$env:LIB"
$env:ROCM_HOME = $rocmCore
$env:ROCM_PATH = $rocmCore
$env:HIP_PATH = $rocmCore
$env:HIP_DEVICE_LIB_PATH = "$rocmCore\lib\llvm\amdgcn\bitcode"
$env:PYTORCH_ROCM_ARCH = "gfx1151"
$env:GPU_ARCHS = "gfx1151"
$env:MAX_JOBS = "32"
$env:CMAKE_BUILD_PARALLEL_LEVEL = "32"
$env:PYTHONPATH = "C:\ComfyUI"
$env:FLASH_ATTENTION_TRITON_AMD_ENABLE = "FALSE"
$env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1"
$env:TORCH_ROCM_FA_PREFER_CK = "0"

$python = "C:\Users\HarutoWatanabe\AppData\Local\Programs\Python\Python313\python.exe"
& $python "$PSScriptRoot\bench_minimax_h3_gfx1151_attention.py" @args --require-aotriton
exit $LASTEXITCODE
