# Define arrays of literal values
$nc_array = 128, 192, 256, 320, 384
$kc_array = 256, 320, 384, 448, 512, 578, 640

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
        
        # echo nc and kc to out.txt
        "nc: $env:GLARE_NC, kc: $env:GLARE_KC" | Out-File -Append -FilePath out.txt
        & ./target/release/bench --m 6000 --n 6000 --k 6000 --t-layout nt --bench-type sgemm --backend glare | Out-File -Append -FilePath out.txt
        Start-Sleep -Seconds 1
    }
}