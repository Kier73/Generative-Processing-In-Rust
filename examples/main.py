from vgpu import vGPU, vVRAM
import time

def benchmark_vgpu():
    gpu = vGPU(sm_count=4)
    data = [1.0, 2.0, 3.0, 4.0]

    print("Executing Command 1 (Cold Start)...")
    start = time.perf_counter_ns()
    res1 = gpu.submit_command("EXEC_VERTEX", data)
    end = time.perf_counter_ns()
    print(f"Cold Time: {end - start} ns")

    print("\nExecuting Command 2 (Warm Start - Recall)...")
    start = time.perf_counter_ns()
    res2 = gpu.submit_command("EXEC_VERTEX", data)
    end = time.perf_counter_ns()
    print(f"Warm Time: {end - start} ns")

    gpu.stats()

def test_memory():
    print("\n--- Memory Substrate Test ---")
    vram = vVRAM()
    addr = (1, 2, 3, 4)
    print(f"Generative Read at {addr}: {vram.read(addr)}")
    vram.write(addr, 0.99)
    print(f"Explicit Read at {addr}: {vram.read(addr)}")

if __name__ == "__main__":
    benchmark_vgpu()
    test_memory()
