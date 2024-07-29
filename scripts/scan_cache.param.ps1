# Define arrays of literal values
$nc_array = 128, 192, 256, 320, 320, 448, 512, 578, 640, 768, 960
$kc_array = 128, 192, 256, 320, 320, 448, 512, 578, 640, 768, 960

$mr = 24
$nr = 4

$env:CORENUM_SGEMM_MR = $mr
$env:CORENUM_SGEMM_NR = $nr

# Outer loop
foreach ($n_i in $nc_array) {
    # Inner loop
    foreach ($k_i in $kc_array) {
        # Set environment variables NC and KC
        $env:NC = $n_i
        $env:KC = $k_i
        
        # echo nc and kc to out.txt
        "nc: $env:NC, kc: $env:KC" | Out-File -Append -FilePath out.txt
        & ./target/release/bench --m 6000 --n 6000 --k 6000 --t-layout nt | Out-File -Append -FilePath out.txt
        Start-Sleep -Seconds 1
    }
}