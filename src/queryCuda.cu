#include <cuda_runtime.h>
#include <iostream>

int main() {
    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    std::cout << "Shared Memory Per Block: " << sharedMemPerBlock << " bytes" << std::endl;
    return 0;
}
