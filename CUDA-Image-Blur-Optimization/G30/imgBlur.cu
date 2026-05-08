#include "libwb/wb.h"
#include "my_timer.h"

#define wbCheck(stmt)							\
  do {									\
    cudaError_t err = stmt;						\
    if (err != cudaSuccess) {						\
      wbLog(ERROR, "Failed to run stmt ", #stmt);			\
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));	\
      return -1;							\
    }									\
  } while (0)

#define BLUR_SIZE 21

///////////////////////////////////////////////////////
#define TILE_WIDTH 16
#define SHARED_WIDTH (TILE_WIDTH + 2 * BLUR_SIZE)
float *deviceTempImageData;

//naive (128 seconds Sequential calkculation)
// __global__ void blurKernel(float *out, float *in, int width, int height) {
//   for(int Row=0; Row < height; Row++){
//     for(int Col=0; Col < width; Col++){
//       float pixVal = 0.0f; 
//       int pixels = 0;

//       for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
//         for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
//           int curRow = Row + blurRow;
//           int curCol = Col + blurCol;
//           // Verify we have a valid image pixel
//           if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
//             pixVal += (float)in[curRow * width + curCol];
//             pixels++;
//           }
//         }
//       }
//       out[Row * width + Col] = (pixVal / (float)pixels);
//     }
//   }
// }


// Getting the correct output (Part A of Q1)
// __global__ void blurKernel(float *out, float *in, int width, int height) {
//   int Col = blockIdx.x * blockDim.x + threadIdx.x;
//   int Row = blockIdx.y * blockDim.y + threadIdx.y;
//   if (Col < width && Row < height) {
//     float pixVal = 0.0f; int pixels = 0;
//     // Get the average of the surrounding 2xBLUR_SIZE x 2xBLUR_SIZE box
//     for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow) {
//       for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
//         int curRow = Row + blurRow;
//         int curCol = Col + blurCol;
//         // Verify we have a valid image pixel
//         if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
//           pixVal += (float)in[curRow * width + curCol];
//           // Keep track of number of pixels in the accumulated total
//           pixels++;
//         }
//       }
//     }
//     // Write our new average pixel value out
//     out[Row * width + Col] = (pixVal / (float)pixels);
//     }
// }

// This does 0.1365 seconds only algorithm
// __global__ void blurKernel(float *out, float *in, int width, int height) {
//   __shared__ float ds_in[SHARED_WIDTH][SHARED_WIDTH];

//   int tx = threadIdx.x;
//   int ty = threadIdx.y;
//   int Col = blockIdx.x * TILE_WIDTH + tx;
//   int Row = blockIdx.y * TILE_WIDTH + ty;

//   // Load into shared memory (halo included)
//   for(int i = ty; i < SHARED_WIDTH; i += TILE_WIDTH) {
//     for(int j = tx; j < SHARED_WIDTH; j += TILE_WIDTH) {
//       int currRow = blockIdx.y * TILE_WIDTH - BLUR_SIZE + i;
//       int currCol = blockIdx.x * TILE_WIDTH - BLUR_SIZE + j;
//       if(currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
//         ds_in[i][j] = in[currRow * width + currCol];
//       } else {
//         ds_in[i][j] = 0.0f;
//       }
//     }
//   }

//   __syncthreads();

//   if (Col < width && Row < height) {
//     int minBlurRow = max(-BLUR_SIZE, -Row);
//     int maxBlurRow = min(BLUR_SIZE, height - 1 - Row);
//     int minBlurCol = max(-BLUR_SIZE, -Col);
//     int maxBlurCol = min(BLUR_SIZE, width - 1 - Col);

//     float pixVal = 0.0f;
//     int pixels = (maxBlurRow - minBlurRow + 1) * (maxBlurCol - minBlurCol + 1);

//     for (int blurRow = minBlurRow; blurRow <= maxBlurRow; ++blurRow) {
//       int sharedRow = ty + BLUR_SIZE + blurRow;
//       for (int blurCol = minBlurCol; blurCol <= maxBlurCol; ++blurCol) {
//         int sharedCol = tx + BLUR_SIZE + blurCol;
//         pixVal += ds_in[sharedRow][sharedCol];
//       }
//     }

//     out[Row * width + Col] = pixVal / pixels;
//   }
// }


__global__ void blurKernelHorizontal(float *temp, const float *in, int width, int height) {
  __shared__ float ds_in[TILE_WIDTH][SHARED_WIDTH + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tileColBase = blockIdx.x * TILE_WIDTH;
  const int Row = blockIdx.y * TILE_WIDTH + ty;
  const int Col = tileColBase + tx;

  for (int j = tx; j < SHARED_WIDTH; j += TILE_WIDTH) {
    const int curCol = tileColBase - BLUR_SIZE + j;
    if (Row < height && curCol >= 0 && curCol < width) {
      ds_in[ty][j] = in[Row * width + curCol];
    } else {
      ds_in[ty][j] = 0.0f;
    }
  }

  __syncthreads();

  if (tx == 0) {
    float running = ds_in[ty][0];
    for (int j = 1; j < SHARED_WIDTH; ++j) {
      running += ds_in[ty][j];
      ds_in[ty][j] = running;
    }
  }

  __syncthreads();

  if (Row < height && Col < width) {
    const int left = tx;
    const int right = tx + (2 * BLUR_SIZE);
    const float leftPrefix = (left > 0) ? ds_in[ty][left - 1] : 0.0f;
    temp[Row * width + Col] = ds_in[ty][right] - leftPrefix;
  }
}

__global__ void blurKernelVertical(float *out, const float *temp, int width, int height) {
  __shared__ float ds_in[SHARED_WIDTH][TILE_WIDTH + 1];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tileRowBase = blockIdx.y * TILE_WIDTH;
  const int Col = blockIdx.x * TILE_WIDTH + tx;
  const int Row = tileRowBase + ty;

  for (int i = ty; i < SHARED_WIDTH; i += TILE_WIDTH) {
    const int curRow = tileRowBase - BLUR_SIZE + i;
    if (curRow >= 0 && curRow < height && Col < width) {
      ds_in[i][tx] = temp[curRow * width + Col];
    } else {
      ds_in[i][tx] = 0.0f;
    }
  }

  __syncthreads();

  if (ty == 0) {
    float running = ds_in[0][tx];
    for (int i = 1; i < SHARED_WIDTH; ++i) {
      running += ds_in[i][tx];
      ds_in[i][tx] = running;
    }
  }

  __syncthreads();

  if (Col < width && Row < height) {
    const int top = ty;
    const int bottom = ty + (2 * BLUR_SIZE);
    const float topPrefix = (top > 0) ? ds_in[top - 1][tx] : 0.0f;
    const float sum = ds_in[bottom][tx] - topPrefix;

    const int validRows = min(height - 1, Row + BLUR_SIZE) - max(0, Row - BLUR_SIZE) + 1;
    const int validCols = min(width - 1, Col + BLUR_SIZE) - max(0, Col - BLUR_SIZE) + 1;
    out[Row * width + Col] = sum / (validRows * validCols);
  }
}
///////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  wbImage_t goldImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *goldOutputImageData;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);

  char *goldImageFile = argv[2];
  goldImage = wbImport(goldImageFile);

  // The input image is in grayscale, so the number of channels is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  goldOutputImageData = wbImage_getData(goldImage);

  // Start timer
  timespec timer = tic();
  ////////////////////////////////////////////////
  //@@ INSERT AND UPDATE YOUR CODE HERE

  dim3 blockSize(16, 16, 1);
  dim3 gridSize((imageWidth + blockSize.x - 1) / blockSize.x, (imageHeight + blockSize.y - 1) / blockSize.y, 1);  // Allocate cuda memory for device input and ouput image data

  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceTempImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice);

  // Call your GPU kernel 10 times
  for(int i = 0; i < 10; i++)
  {
    // // Use this for the Horizontal and Vertical pass
    blurKernelHorizontal<<<gridSize, blockSize>>>(deviceTempImageData,
                                                  deviceInputImageData,
                                                  imageWidth,
                                                  imageHeight);
    blurKernelVertical<<<gridSize, blockSize>>>(deviceOutputImageData,
                                                deviceTempImageData,
                                                imageWidth,
                                                imageHeight);

    // blurKernel<<<gridSize, blockSize>>>(deviceOutputImageData,
    //                               deviceInputImageData, imageWidth,
    //                               imageHeight);
  }
  cudaDeviceSynchronize();
  // Transfer data from GPU to CPU
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(deviceTempImageData);
  ///////////////////////////////////////////////////////
  
  // Stop and print timer
  toc(&timer, "GPU execution time (including data transfer) in seconds");
  // Check the correctness of your solution
  // wbSolution(args, outputImage);

   for(int i=0; i<imageHeight; i++){
     for(int j=0; j<imageWidth; j++){
       if(abs(hostOutputImageData[i*imageWidth+j]-goldOutputImageData[i*imageWidth+j])/goldOutputImageData[i*imageWidth+j]>0.01){
         printf("Incorrect output image at pixel (%d, %d): goldOutputImage = %f, hostOutputImage = %f\n", i, j, goldOutputImageData[i*imageWidth+j],hostOutputImageData[i*imageWidth+j]);
	 return -1;
       }
     }
   }
   printf("Correct output image!\n");

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
