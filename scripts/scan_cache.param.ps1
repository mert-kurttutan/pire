    # Define arrays of literal values
$nc_array = 56, 112, 168
$kc_array = 192, 256, 320, 384, 512

$mr = 24
$nr = 4

$env:PIRE_SGEMM_MR = $mr
$env:PIRE_SGEMM_NR = $nr

# Outer loop
foreach ($n_i in $nc_array) {
    # Inner loop
    foreach ($k_i in $kc_array) {
        # Set environment variables NC and KC
        $env:PIRE_NC = $n_i
        $env:PIRE_KC = $k_i
        
        # echo nc and kc to out.txt
        "nc: $env:PIRE_NC, kc: $env:PIRE_KC" | Out-File -Append -FilePath out.txt
        & cargo bench --features "mkl" | Out-File -Append -FilePath out.txt
        Start-Sleep -Seconds 1
    }
}