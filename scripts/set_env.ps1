# Define and set environment variables
$env_dir = $args[0]

$env:GLAR_MKL_PATH = "$env_dir\.env\Library\bin"
$env:Path += ";" + $env:GLAR_MKL_PATH
# $env:GLAR_SGEMM_NC = 
# $env:GLAR_SGEMM_KC = 
# $env:GLAR_SGEMM_MR =
# $env:GLAR_SGEMM_NR =

