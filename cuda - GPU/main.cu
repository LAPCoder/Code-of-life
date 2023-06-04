#include <iostream>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define YEAR_DURATION  std::chrono::milliseconds{100}
#define START_COUNT_T  50
#define VALIDITY_CHECK 0xB0B0

struct t_o_l
{
	explicit t_o_l(): max_age{(unsigned char)(rand() % 140 + 10),
							  (unsigned char)(rand() % 140 + 10)} {};
	explicit t_o_l(unsigned char ma1, unsigned char ma2): max_age{ma1, ma2} {};
	unsigned char age = 0;
	unsigned char max_age[2]; // TODO: opti: store only 1 max_age
	unsigned char hp = 100;
	bool coupled = false;
	unsigned short validity_check = VALIDITY_CHECK;
};

unsigned NUM_ELEMS = START_COUNT_T;

std::ostream &operator<<(std::ostream &a, t_o_l *b)
{
	for (unsigned i = 0; i < NUM_ELEMS; i++)
		a << std::to_string((int)b[i].age) << ", "
		<< std::to_string(min((int)b[i].max_age[0], (int)b[i].max_age[1]))
		<< ", 0x" << std::hex << std::uppercase << b[i].validity_check
		<< std::nouppercase << std::dec << "; ";
	return a;
}

__global__ void thread_of_live(t_o_l *ALL, unsigned *NUM_ELEMS2, bool *TO_DELETE)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("tid: %u\n", tid);
	if (tid > *NUM_ELEMS2)
		return;

	ALL[tid].age++;

	if (ALL[tid].age >= min((int)ALL[tid].max_age[0], (int)ALL[tid].max_age[1]))
	{
		(*NUM_ELEMS2)--;
		TO_DELETE[tid] = true;
	}
	//else printf("Alive!");
}

int main()
{
	t_o_l *ALL, *d_ALL;
	unsigned *d_NUM_ELEMS;
	bool *TO_DELETE, *d_TO_DELETE;

	ALL = (t_o_l*)malloc(sizeof(t_o_l) * START_COUNT_T);
	TO_DELETE = (bool*)malloc(sizeof(bool) * START_COUNT_T);
	assert(ALL != NULL);
	assert(TO_DELETE != NULL);

	for (int i = 0; i < START_COUNT_T; i++)
	{
		new (ALL + i) t_o_l; // This line calls constructors.
		TO_DELETE[i] = false;
	}

	//std::cout << ALL << std::endl;
	
	cudaMalloc((void**)&d_ALL,		sizeof(t_o_l) * START_COUNT_T);
	cudaMalloc((void**)&d_NUM_ELEMS,sizeof(unsigned));
	cudaMalloc((void**)&d_TO_DELETE,sizeof(bool));
	cudaMemcpy(d_ALL,		ALL,		sizeof(t_o_l)*START_COUNT_T,cudaMemcpyHostToDevice);
	cudaMemcpy(d_NUM_ELEMS,	&NUM_ELEMS,	sizeof(unsigned),			cudaMemcpyHostToDevice);
	cudaMemcpy(d_TO_DELETE,	TO_DELETE,	sizeof(bool)*START_COUNT_T,	cudaMemcpyHostToDevice);

	thread_of_live<<<1,256>>>(d_ALL, d_NUM_ELEMS, d_TO_DELETE);

	while (NUM_ELEMS != 0)
	{
		std::cout << NUM_ELEMS << " threads needed, "
			<< std::ceil(((float)NUM_ELEMS)/256)*256 << " threads launched ("
			<< std::ceil(((float)NUM_ELEMS)/256) << " blocks)." << ALL << std::endl;

		cudaMemcpy(TO_DELETE,	d_TO_DELETE,sizeof(bool)*NUM_ELEMS,		cudaMemcpyDeviceToHost); // Must be before cpy NUM_ELEMS
		auto tmp = NUM_ELEMS;
		cudaMemcpy(&NUM_ELEMS,	d_NUM_ELEMS,sizeof(unsigned),			cudaMemcpyDeviceToHost);
		cudaMemcpy(ALL,			d_ALL,		sizeof(t_o_l) * NUM_ELEMS,	cudaMemcpyDeviceToHost);
		if (tmp != NUM_ELEMS)
			for (int i = tmp-1; i > 0; i--)
				if (TO_DELETE[i] == true) // Yes, but to be sure...
				{
					for(int j = i; j < tmp-1; j++)
						ALL[j] = ALL[j+1];
					TO_DELETE[i] = false;
				}
		if (tmp == NUM_ELEMS) printf("\nWORK!!!\n"); // TODO: test (and report bug?): i can't write 'else' (consider it's on the 'for' loop)
		ALL = (t_o_l*)malloc(sizeof(t_o_l) * NUM_ELEMS);
		assert(ALL != NULL);

		// Validity check
		for (unsigned i = 0; i < NUM_ELEMS; i++)
			if (ALL[NUM_ELEMS].validity_check != VALIDITY_CHECK)
			{
				printf("Error: Bad memory transfer. Element: %u, value: %u, tmp=%u, NUM_ELEMS=%i",
						i, ALL[NUM_ELEMS].validity_check, tmp, NUM_ELEMS);
				exit(-1);
			}

		cudaMalloc((void**)&d_ALL, sizeof(t_o_l) * NUM_ELEMS);
		cudaMemcpy(d_ALL, ALL, sizeof(t_o_l)*NUM_ELEMS, cudaMemcpyHostToDevice);
		
		thread_of_live<<<std::ceil(((float)NUM_ELEMS)/256),256>>>(d_ALL, d_NUM_ELEMS, d_TO_DELETE);
		cudaDeviceSynchronize();
	}

	for (int i = 0; i < NUM_ELEMS; i++)
		ALL[i].~t_o_l(); // This line calls destructors.

	free(ALL);

	return 0;
}
