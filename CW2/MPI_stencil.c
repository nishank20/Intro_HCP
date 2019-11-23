#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"
#include <string.h>
// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#ifdef UNROLL
#define min(x, y) ((x) < (y) ? (x) : (y))
#endif
void stencil(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void topStencil(const int nx, const int ny, const int width, const int height,
                float * image, float * tmp_image);
void bottomStencil(const int nx, const int ny, const int width, const int height,
                float * image, float * tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);

int main(int argc, char* argv[])
{
  int flag;
  int size;
  int rank;
  MPI_Status status;
  MPI_Init(&argc, &argv);

  MPI_Initialized(&flag);
  if (flag != 1 ) {
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // Check usage
  //if (argc != 4) {
  //  fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
  //  exit(EXIT_FAILURE);
  //}
  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);
  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil
  int width = nx + 2;
  int height = ny + 2;
  float out_image[width*height];

  // Allocate the image
  //double* image = malloc(sizeof(double) * width * height);
  //double* tmp_image = malloc(sizeof(double) * width * height);
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(float) * width * height);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);
  // float *send_buf1= malloc(sizeof(float)*ny);
  float up_buf[height];
  float down_buf[height];

  // 2. try to run the stencil by one master and 4 workers

  // double gtic = MPI_Wtime();
  int numCores = size;
  int height_1 = width/numCores;
  int height_2 = nx/numCores;
  // float *localImg = malloc(sizeof(float)*ny*(nx/numCores+2));
  float localImg[height*(width/numCores+2)];

  float localTmp[height*(width/numCores+1)];
  float localImg0[height*(width/numCores+1)];

  // float localImg1[ny*(nx/numCores+2)];
  // float localImg2[ny*(nx/numCores+2)];


  int start = rank * height_1;

  if (size == 1){
    double tic = wtime();
    for (int i =0; i < niters; i++){
      stencil(nx, ny, width, height, image, tmp_image);
      stencil(nx, ny, width, height, tmp_image, image);
    }
    double toc = wtime();
    printf("------------------------------------\n");
    printf(" runtime for one worker: %lf s\n", toc-tic);
    printf("------------------------------------\n");
  }

  else if (rank == 0) {

    for (int i = 0; i < height_1; ++i) {
      for (int j = 0; j < ny; ++j) {
        localImg0[j+i*height] = image[j+i*height];
      }
    }
    double tic = wtime();

    for (int i =0; i < niters; i++){

      // send neighbours wanted rows
      for (int j = 0; j < height; ++j) {
        down_buf[j] = localImg0[j+(height_1-1)*height];
      }
      MPI_Ssend(&down_buf, height, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);

      //get rows from neighbour rank
      MPI_Recv(&up_buf, height, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
      for (int j = 0; j < height; ++j) {
        localImg0[j+height_1*height] = up_buf[j];
      }
      // send_recv(ny, height, send_buf1, send_buf2, localImg, status, 1, 1, end);
      topStencil(height_2+1, ny,height_1+1, height, localImg0, localTmp);

      //send neighbours wanted rows
      for (int j = 0; j < ny; ++j) {
        down_buf[j] = localTmp[j+(height_1-1)*height];
      }
      MPI_Ssend(&down_buf, height, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);

      MPI_Recv(&up_buf, height, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
      //get rows from neighbour rank
      for (int j = 0; j < height; ++j) {
        localTmp[j+height_1*height] = up_buf[j];
      }
      // send_recv(ny, height, send_buf1, send_buf2, localTmp, status, 1, 1, end);
      topStencil(height_2+1, ny,height_1+1, height, localTmp, localImg0);
    }

    double toc = wtime();
    printf("------------------------------------\n");
    printf(" runtime rank0: %lf s\n", toc-tic);
    printf("------------------------------------\n");
    output_image("stencil0.pgm",height_2+1, ny, height_1+1, height, localImg0);
    for (int i = 0; i < height_1; ++i) {
      for (int j = 0; j < height; ++j) {
        out_image[j+i*height] = localImg0[j+i*height];
      }
    }
    printf ("%f to %f from rank %d\n", localImg0[0], localImg0[(height-1)+(height_1)*height], 0);

    free(localImg0);

    MPI_Recv(localImg0, height*(width/numCores+1), MPI_FLOAT, size-1, 0, MPI_COMM_WORLD, &status);
    printf ("%f to %f from rank %d\n", localImg0[height], localImg0[(height-1)+height_1*height], size-1);
    for (int i = 0; i < height_1; ++i) {
        for (int j = 0; j < height; ++j) {
          out_image[j+((size-1)*height_1+i)*height] = localImg0[j+(i+1)*ny];
        }
    }
    printf("%f to %f given by rank %d\n", out_image[((size-1)*height_1)*height], out_image[height-1+((size-1)*height_1+height_1-1)*height], size-1);

    printf ("%f \n", out_image[((size-1)*height_1)*height]);
    for (int src = 1; src < size-1; src++){

      MPI_Recv(localImg, ny*(width/numCores+2), MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
      printf ("%f to %f from rank %d\n", localImg[ny], localImg[(height-1)+(height_1)*ny], src);
      for (int i = 0; i < height_1; ++i) {
        for (int j = 0; j < height; ++j) {
          out_image[j+(src*height_1+i)*height] = localImg[j+(i+1)*height];
        }

      }
      printf("%f to %f given by rank %d\n", out_image[(src*height_1)*height], out_image[height-1+(src*height_1+height_1-1)*height], src);

    }
    // printf ("%f to %f \n", out_image[0], out_image[(ny-1)/2]);
    output_image(OUTPUT_FILE, nx, ny, width, height, out_image);
    // free(out_image);
  }

  else if (rank==size-1){
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < ny; ++j) {
        localImg0[j+(i+1)*ny] = image[j+(start+i)*ny];
      }
    }
    double tic = wtime();
    if(rank % 2 == 0){
      printf("even\n");

      for (int i =0; i < niters; i++){

        //send lower rank the first row
        for (int j = 0; j < ny; ++j) {
          up_buf[j] = localImg0[j+ny];
        }
        MPI_Ssend(&up_buf, height, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);

        //receive from lower rank its last row
        MPI_Recv(&down_buf, height, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < ny; ++j) {
          localImg0[j] = down_buf[j];
        }

        bottomStencil(height_1+1, ny,height_1+1, height, localImg0, localTmp);

        //send lower rank the first row
        for (int j = 0; j < ny; ++j) {
          up_buf[j] = localTmp[j+ny];
        }
        MPI_Ssend(&up_buf, height, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
        //receive from lower rank its last row
        MPI_Recv(&down_buf, height, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < ny; ++j) {
          localTmp[j] = down_buf[j];
        }

        bottomStencil(height_1+1, ny,height_1+1, height, localTmp, localImg0);

      }
    double toc = wtime();

    printf("------------------------------------\n");
    printf(" runtime rank %d: %lf s\n", rank, toc-tic);
    printf("------------------------------------\n");
    MPI_Ssend(localImg0, height*(width/numCores+1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    output_image("stencil3.pgm", height_1+1, ny,height_1+1, height, localImg0);

    }
    else if (rank % 2 == 1){
      printf("odd\n");
      double tic = wtime();
      for (int i =0; i < niters; i++){
        //receive from lower rank its last row
        MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < ny; ++j) {
          localImg0[j] = down_buf[j];
        }
        //send lower rank the first row
        for (int j = 0; j < ny; ++j) {
          up_buf[j] = localImg0[j+ny];
        }
        MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);

        bottomStencil(height_1+1, ny,height_1+1, height, localImg0, localTmp);

        //receive from lower rank its last row
        MPI_Recv(&down_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
        for (int j = 0; j < ny; ++j) {
          localTmp[j] = down_buf[j];
        }
        //send lower rank the first row
        for (int j = 0; j < ny; ++j) {
          up_buf[j] = localTmp[j+ny];
        }
        MPI_Ssend(&up_buf, ny, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);

        bottomStencil(height_1+1, ny,height_1+1, height, localTmp, localImg0);

      }
    double toc = wtime();
    printf("------------------------------------\n");
    printf(" runtime rank %d: %lf s\n", rank, toc-tic);
    printf("------------------------------------\n");
    MPI_Ssend(localImg0, height*(width/numCores+1), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    output_image("stencil3.pgm", height_1+1, ny, height_1+1, height, localImg0);

    }
  }


  MPI_Finalize();

  return EXIT_SUCCESS;
}
void send_recv(int ny, int height, float send_buf[], float recv_buf[], float img[], MPI_Status status, int src, int des, int end){
  for (int j = 0; j < ny; ++j) {
    send_buf[j] = img[j+end*ny];
  }
  MPI_Ssend(&send_buf, ny, MPI_FLOAT, des, 0, MPI_COMM_WORLD);
  MPI_Recv(&recv_buf, ny, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
  //get rows from neighbour rank
  for (int j = 0; j < ny; ++j) {
    img[j+height*ny] = recv_buf[j];
  }
}

/*
void stencil(const int nx, const int ny, const int width, const int height,
             double* image, double* tmp_image)
{
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      tmp_image[j + i * height] =  image[j     + i       * height] * 3.0 / 5.0;
      tmp_image[j + i * height] += image[j     + (i - 1) * height] * 0.5 / 5.0;
      tmp_image[j + i * height] += image[j     + (i + 1) * height] * 0.5 / 5.0;
      tmp_image[j + i * height] += image[j - 1 + i       * height] * 0.5 / 5.0;
      tmp_image[j + i * height] += image[j + 1 + i       * height] * 0.5 / 5.0;
    }
  }
}
*/
void topStencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  #ifndef UNROLL
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      tmp_image[i + j * height] = (image[i + j * height] * 0.6f +(image[i + (j - 1) * height] + image[i + (j + 1) * height] +image[(i - 1) + j * height] + image[(i + 1) + j * height]) *0.1f);
    }
  }
#endif
}
void bottomStencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  #ifndef UNROLL
  #else
    const int ny_1 = ny + 1;
    const int nx_1 = nx + 1;
    for (int j = 1; j < ny_1; j += 1024) {
      for (int i = 1; i < nx_1; i += 1024) {
        const int jm = min(j + 1024, ny1);
        for (int jj = j; jj < jm; ++jj) {
          const int im = min(i + 1024, nx1);
          for (int ii = i; ii < im; ++ii) {
                  tmp_image[ii + jj * width] =(image[ii + jj * width] * 0.6f +(image[ii + (jj - 1) * width] + image[ii + (jj + 1) * width] +image[(ii - 1) + jj * width] + image[(ii + 1) + jj * width])
                                                  * 0.1f);
          }
        }
      }
    }
    #endif
}
void stencil(const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  #ifndef UNROLL
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      tmp_image[i + j * height] = (image[i + j * height] * 0.6f +(image[i + (j - 1) * height] + image[i + (j + 1) * height] +image[(i - 1) + j * height] + image[(i + 1) + j * height]) *0.1f);
    }
  }
#else
  const int ny_1 = ny + 1;
  const int nx_1 = nx + 1;
  for (int j = 1; j < ny_1; j += 1024) {
    for (int i = 1; i < nx_1; i += 1024) {
      const int jm = min(j + 1024, ny1);
      for (int jj = j; jj < jm; ++jj) {
        const int im = min(i + 1024, nx1);
        for (int ii = i; ii < im; ++ii) {
                tmp_image[ii + jj * width] =(image[ii + jj * width] * 0.6f +(image[ii + (jj - 1) * width] + image[ii + (jj + 1) * width] +image[(ii - 1) + jj * width] + image[(ii + 1) + jj * width])
                                                * 0.1f);
        }
      }
    }
  }
#endif
}
// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
