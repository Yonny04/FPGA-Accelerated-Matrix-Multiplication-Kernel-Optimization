#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "kernel_gemm.h"
#include "xcl2.hpp"

#ifndef ENABLE_VERIFY
#define ENABLE_VERIFY 1
#endif

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <xclbin>\n";
    return 1;
  }
  std::string binaryFile = argv[1];

  //the array sizes for A, B and C
  const size_t sizeA = (size_t)NI * (size_t)NK;
  const size_t sizeB = (size_t)NK * (size_t)NJ;
  const size_t sizeC = (size_t)NI * (size_t)NJ;

  /*this is the part that i needed help with *C0* */
  std::vector<float, aligned_allocator<float>> A(sizeA);
  std::vector<float, aligned_allocator<float>> B(sizeB);
  std::vector<float, aligned_allocator<float>> C(sizeC);
  //C0 is the original C before being updated by the kernel
  std::vector<float, aligned_allocator<float>> C0(sizeC);

  //chekcing with random values for A, B and C
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> distf(-1.0f, 1.0f);

  //assigning the random values to the A, B and C array
  for (size_t i = 0; i < sizeA; ++i) A[i] = distf(rng);
  for (size_t i = 0; i < sizeB; ++i) B[i] = distf(rng);
  //use C0 to keep a copy of the original C for verification
  //C gets instanly updated by the kernel, so S0 is needed for CPU reference
  for (size_t i = 0; i < sizeC; ++i) {
    float v = distf(rng);
    //initialize C and C0 with the same values
    C[i]  = v;
    C0[i] = v;
  }
  //alpha and beta values
  const float alpha = 1.5f;
  const float beta  = 2.5f;

  // OpenCL/XRT setup
  cl_int err = CL_SUCCESS;
  auto devices = xcl::get_xil_devices();
  cl::Device device = devices[0];

  cl::Context context(device, nullptr, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create context, err=" << err << "\n";
    return 1;
  }

  cl::CommandQueue q(context, device, 0, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create command queue, err=" << err << "\n";
    return 1;
  }
  
  auto fileBuf = xcl::read_binary_file(binaryFile);
  cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};

  cl::Program program(context, {device}, bins, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to program device with xclbin, err=" << err << "\n";
    return 1;
  }

  cl::Kernel kernel(program, "kernel_gemm", &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to create kernel 'kernel_gemm', err=" << err << "\n";
    return 1;
  }

  // Buffers
  //Buffer for C, with read and write access
  //the kernel reads the original C values, updates them and writes them back to C
  //the writing is because of the store_C function that writes the updated C values
  cl::Buffer bufferC(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                     sizeof(float) * sizeC, C.data(), &err);
  //checking if the data alocation for buffer C was successful
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to allocate bufferC, err=" << err << "\n";
    return 1;
  }
  //Buffers for A and B both with read only access
  cl::Buffer bufferA(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(float) * sizeA, A.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to allocate bufferA, err=" << err << "\n";
    return 1;
  }

  cl::Buffer bufferB(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                     sizeof(float) * sizeB, B.data(), &err);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to allocate bufferB, err=" << err << "\n";
    return 1;
  }

  // kernel_gemm(C, A, B, alpha, beta)
  //loading the buffers, alpha and beta for kernel_gemm computation
  err  = kernel.setArg(0, bufferC);
  err |= kernel.setArg(1, bufferA);
  err |= kernel.setArg(2, bufferB);
  err |= kernel.setArg(3, alpha);
  err |= kernel.setArg(4, beta);
  //cheking if the kernel arguments were set successfully
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to set kernel args, err=" << err << "\n";
    return 1;
  }


  //migrating the data from the host to the device
  err = q.enqueueMigrateMemObjects({bufferA, bufferB, bufferC}, 0);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to migrate inputs, err=" << err << "\n";
    return 1;
  }
  q.finish();

  //enqueue the kernel for execution
  //running the kernel
  err = q.enqueueTask(kernel);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to enqueue kernel task, err=" << err << "\n";
    return 1;
  }
  q.finish();

  //migrating the updated C values back from the device to the host
  err = q.enqueueMigrateMemObjects({bufferC}, CL_MIGRATE_MEM_OBJECT_HOST);
  if (err != CL_SUCCESS) {
    std::cerr << "Failed to migrate outputs, err=" << err << "\n";
    return 1;
  }
  q.finish();

  //testing the kernel output by comparing it to the CPU reference implementation
  //comparing the kernel C output with the CPU reference C0 output
#if ENABLE_VERIFY
  //setting the test size for verification and the tolerance
  const int i_test = 8;
  const int j_test = 8;
  const float tolerance = 1e-2f;
  int errors = 0;

  for (int i = 0; i < i_test; ++i) {
    for (int j = 0; j < j_test; ++j) {
      float sum = 0.0;
      for (int k = 0; k < NK; ++k) {
        sum += (float)A[(size_t)i * NK + k] * (float)B[(size_t)k * NJ + j];
      }
      float expected_result = beta * C0[(size_t)i * NJ + j] + alpha * (float)sum;
      float actual_result  = C[(size_t)i * NJ + j];
      float difference = std::fabs(actual_result - expected_result);
     
      if (difference > tolerance) {
        errors++;
      }
    }
  }

  if (errors == 0) {
    std::cout << "PASS\n";
    return 0;
  } 
  else {
    std::cout << "FAIL: " << errors << " mismatches\n";
    return 1;
  }

#else
  //also do checksum for a quick output check
  float checksum = 0.0;
  for (size_t idx = 0; idx < sizeC; ++idx){
    checksum += (float)C[idx];
    std::cout << "Checksum(C) = " << checksum << "\n";
    return 0;  
    } so
#endif
}
