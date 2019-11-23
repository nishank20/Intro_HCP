#stencil: stencil.c
#	gcc -std=c99 -Wall $^ -o $@
CC=mpicc

COMP=GNU
ifeq ($(COMP), GNU)
  CFLAGS=-Wall -fopenmp-simd -Ofast -march=native
endif

#EXE1=mpi_stencil
EXES=MPI_stencil

#all: $(EXES)

mpi_stencil: MPI_stencil.c
#$(EXES): %.exe : %.c
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: clean all

clean:
	\rm -f $(EXES)
	\rm -f *.o
	\rm machine.file*
	\rm MPI.*
	\rm mpi_stencil.80s*
