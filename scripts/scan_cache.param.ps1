    # Define arrays of literal values
$nc_array = 112, 168, 224
$kc_array = 192, 256, 320, 384, 512

$mr = 24
$nr = 4

$env:GLARE_SGEMM_MR = $mr
$env:GLARE_SGEMM_NR = $nr

# Outer loop
foreach ($n_i in $nc_array) {
    # Inner loop
    foreach ($k_i in $kc_array) {
        # Set environment variables NC and KC
        $env:GLARE_NC = $n_i
        $env:GLARE_KC = $k_i
        
        # echo nc and kc to out2.txt
        "nc: $env:GLARE_NC, kc: $env:GLARE_KC" | Out-File -Append -FilePath out2.txt
        & cargo bench --features "mkl" | Out-File -Append -FilePath out2.txt
        Start-Sleep -Seconds 1
    }
}