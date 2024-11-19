# Define and set environment variables
$env_dir = $args[0]

$env:PIRE_MKL_PATH = "$env_dir\.env\Library\bin"
$env:Path += ";" + $env:PIRE_MKL_PATH
# $env:PIRE_SGEMM_NC = 
# $env:PIRE_SGEMM_KC = 
# $env:PIRE_SGEMM_MR =
# $env:PIRE_SGEMM_NR =

