# Define the location for the virtual environment
$env_path =  # Replace with your desired environment path

# Create the virtual environment
python -m venv $env_path

# Activate the virtual environment
& "$env_path\Scripts\Activate.ps1"

# Upgrade pip
pip install --upgrade pip

# Install MKL
pip install mkl

# Array of .dll library names with specific versions
$dll_libraries = @("mkl_rt.2.dll", "mkl_core.2.dll", "mkl_intel_thread.2.dll")

# Directory where the libraries are located
$lib_dir = "$env_path\Library\bin"

# Loop through the array and create symbolic links
foreach ($lib in $dll_libraries) {
    # Extract the base name (without version and .dll extension)
    $base_name = $lib -replace "\.2\.dll$", ""
    # Create the symbolic link
    $link_path = "$lib_dir\$base_name.dll"
    $target_path = "$lib_dir\$lib"
    # New-Item -ItemType SymbolicLink -Path $link_path -Target $target_path
    Copy-Item -Path $target_path -Destination $link_path
    # cmd /c mklink "$link_path" "$target_path"
    Write-Host "Created symbolic link: $link_path -> $target_path"
}
