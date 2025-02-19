
import subprocess, os
def scan_params(
        m: int, n: int, k: int, bench_type: str, output_file: str,
        binary_location: str
    ):
    nc_arr = [192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    kc_arr = [256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]

    for nc in nc_arr:
        for kc in kc_arr:
            # set env PIRE_NC=nc, PIRE_KC=kc
            os.environ['PIRE_NC'] = str(nc)
            os.environ['PIRE_KC'] = str(kc)

            # append print the value of env vars PIRE_NC and PIRE_KC to the output file
            with open(output_file, 'a') as f:
                nc = os.environ['PIRE_NC']
                kc = os.environ['PIRE_KC']
                f.write(f"NC={nc}, KC={kc}\n")

            pire_command_str = f"{binary_location} --m {m} --n {n} --k {k} --bench-type {bench_type} --backend pire"
            mkl_command_str = f"{binary_location} --m {m} --n {n} --k {k} --bench-type {bench_type} --backend mkl"

            # execute the pire and mkl commands and append the output to the output file
            subprocess.run(pire_command_str, shell=True, stdout=open(output_file, 'a'))
            subprocess.run(mkl_command_str, shell=True, stdout=open(output_file, 'a'))



if __name__ == "__main__":
    dim = 2400
    bench_type = "sgemm"
    # get binary location in os agnostic way
    binary_location = os.path.join(os.getcwd(), ".", "target", "release", "bench")
    scan_params(dim, dim, dim, bench_type, "output.txt", binary_location)