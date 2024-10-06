# run scripts/prepare.ps1

$current_directory = Get-Location
# provide current directory as 1st argument
. .\scripts\install_mkl.ps1 $current_directory
# run set_env.ps1
. .\scripts\set_env.ps1 $current_directory
