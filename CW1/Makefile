#stencil: stencil.c
	#gcc -std=c99 -Wall $^ -o $@
stencil: stencil.c
	gcc -Ofast -march=native -mtune=intel -ffast-math -funsafe-math-optimizations -pg -fopenmp -g -o stream.gprof stencil.c -o stencil
.PHONY: icc_1
icc_1: stencil.c config.h
	icc -Ofast -xHost -pg -fopenmp -g -o stream.gprof stencil.c -o stencil
.PHONY: clang_1
clang_1: config.h stencil.c
	gcc -Ofast -march=native -mtune=intel -ffast-math -funsafe-math-optimizations -pg -fopenmp -g -o stream.gprof stencil.c O2_1.s -o stencil
O2_1.bs: stencil.c config.h
	clang_1 -O2 -std=c99 -Wall -fwrapv -pg -fopenmp -g -o stream.gprof stencil.c -S -o O2_1.bs

O2_1.s: O2_1.bs
	head -n -1 O2.bs > O2_1.s
.PHONY: watch_me
watch_me:
	clear; while true; do tput cup 0 0; squeue -u qc15863; sleep 1; done
.PHONY: test
test: stencil
	./stencil 1024 1024 100
	python check.py --ref-stencil-file stencil_1024_1024_100.pgm --stencil-file stencil.pgm
	./stencil 4096 4096 100
	python check.py --ref-stencil-file stencil_4096_4096_100.pgm --stencil-file stencil.pgm
	./stencil 8000 8000 100
	python check.py --ref-stencil-file stencil_8000_8000_100.pgm --stencil-file stencil.pgm
