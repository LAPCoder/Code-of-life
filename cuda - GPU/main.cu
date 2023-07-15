// cd C:\Users\andre\Documents\C++\Code-of-Life
// nvcc "cuda - GPU/main.cu" -o main -std=c++17 -g --compiler-options "-WX"

/**--------------------------------------------------------------------------**/
/**---------------------------INCLUDES---------------------------------------**/
/**--------------------------------------------------------------------------**/
#include <iostream>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <iomanip>
#include <ctime>
#include <thread>
#include <chrono>
#include <limits>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

/**--------------------------------------------------------------------------**/
/**---------------------------MACROS-----------------------------------------**/
/**--------------------------------------------------------------------------**/
#define INT_TO_STD_TIME(x)		std::chrono::microseconds(x)
#define STEADY_CLK_TO_std(x)	std::chrono::duration_cast<std::chrono::microseconds>(x)

/**--------------------------------------------------------------------------**/
/**---------------------------DEFINES----------------------------------------**/
/**--------------------------------------------------------------------------**/
#define YEAR_DURATION_ns_uint	100'000'000UL
#define YEAR_DURATION_std		INT_TO_STD_TIME(YEAR_DURATION_ns_uint/1'000UL)
typedef unsigned				nElemTpe;
#define nElemTpePrefix			"%u" // for printf()
#define START_COUNT_T			((nElemTpe)+50)
#define MAX_TH_IN_GEN			((nElemTpe)+1'048'576)
#define VALIDITY_CHECK			0xB0B0
#define GENERATION_DURATION		25	// years
#define MAX_AGE					150	// years
#define MIN_AGE					10	// years
#define DEBUG					1 // comment to set off; comment for less ram use and best performances.

#ifndef ULLONG_MAX
	#define ULLONG_MAX 0xffffffffffffffffui64
#endif
#ifndef USHRT_MAX
	#define USHRT_MAX 0xffffU
#endif
#ifndef UCHAR_MAX
	#define UCHAR_MAX 0xff
#endif

/**--------------------------------------------------------------------------**/
/**---------------------------DEFINES VERIFICATION---------------------------**/
/**--------------------------------------------------------------------------**/
#if START_COUNT_T > MAX_TH_IN_GEN
	#error The maximum number of individuals per generation must be greater than the number of starting individuals. (START_COUNT_T <= MAX_TH_IN_GEN)
#endif
#if MIN_AGE >= MAX_AGE
	#error The maximum age must be greater than the minimum age. (MIN_AGE < MAX_AGE)
#endif
#if (MAX_TH_IN_GEN & (MAX_TH_IN_GEN - 1)) != 0
	#error MAX_TH_IN_GEN must be a power of 2 and different of 0.
#endif

/**--------------------------------------------------------------------------**/
/**---------------------------STRUCTS----------------------------------------**/
/**--------------------------------------------------------------------------**/
struct t_o_l // t_o_l = thread of life
{
	explicit t_o_l(): max_age{(unsigned char)(rand() % (MAX_AGE-MIN_AGE) + MIN_AGE),
							  (unsigned char)(rand() % (MAX_AGE-MIN_AGE) + MIN_AGE)}
							  {};
	explicit t_o_l(unsigned char ma1, unsigned char ma2): max_age{ma1, ma2} {};
	unsigned char age = 0;
	unsigned char max_age[2]; // TODO: opti: store only 1 max_age on 2
	//unsigned char hp = 100;
	//bool coupled = false;
	#ifdef DEBUG
		unsigned short validity_check = VALIDITY_CHECK;
	#endif
};

/**--------------------------------------------------------------------------**/
/**---------------------------PROTOTYPES-------------------------------------**/
/**--------------------------------------------------------------------------**/
void atexit_clean(void *data);
void clean();
nElemTpe num_elems(nElemTpe *NUM_ELEMS);
//__host__ std::ostream &operator<<(std::ostream&, t_o_l*);
__device__ unsigned get_arch();
__global__ void thread_of_live(t_o_l**,	nElemTpe*, unsigned short*);
void deletor(t_o_l (*)[MAX_TH_IN_GEN], nElemTpe *NUM_ELEMS);
t_o_l **setupHMM(std::vector<std::vector<t_o_l>>&, int);

/**--------------------------------------------------------------------------**/
/**---------------------------GLOBAL VARIABLES-------------------------------**/
/**--------------------------------------------------------------------------**/
unsigned short NUM_GENS = 1;

/**--------------------------------------------------------------------------**/
/**---------------------------FUNCTIONS--------------------------------------**/
/**--------------------------------------------------------------------------**/
/*__host__ std::ostream &operator<<(std::ostream &a, t_o_l *b)
{
	for (unsigned i = 0; i < num_elems(NUM_ELEMS); i++)
		a << std::to_string((int)b[i].age) << ", "
		<< std::to_string(min((int)b[i].max_age[0], (int)b[i].max_age[1]))
		*<< ", 0x" << std::hex << std::uppercase << b[i].validity_check
		<< std::nouppercase << std::dec* << "; ";
	return a;
}*/

static void clean(void)
{
	atexit_clean(NULL);
}

void atexit_clean(void *data)
{
	static void *x;

	if (data) {
		x = data;
		atexit(clean);
	} else {
		free(x);
	}
}

__device__ unsigned get_arch()
{
	#ifdef __CUDA_ARCH__
		return __CUDA_ARCH__;
	#else
		return 0;
	#endif
}

/**
 * @brief The main function that turns on GPU.
 * Each thread calculate the life of one person.
 * 
 * @param ALL The main array, in the VRAM.
 * @param NUM_ELEMS2 The array of the number of threads of each generations, in the VRAM.
 * @param NUM_GENS2 The number of generations, in the VRAM.
 * @return Nothing 
 */
__global__ void thread_of_live(
	t_o_l **ALL,
	nElemTpe *NUM_ELEMS2,
	unsigned short *NUM_GENS2)
{
	unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("tid=%llu ", tid);
	#if MAX_TH_IN_GEN > 1024
		#define DIM_X threadIdx.x
		#define DIM_Y blockIdx.x
	#elif
		#define DIM_X blockIdx.x
		#define DIM_Y threadIdx.x
	#endif
	/* Why that distinction?
	 * (read the commant in the call of the function)
	 * Schema of the grid (I use only x dim) in a GPU:
	 * | Block0  | Block1  | Block2  | ...
	 * +---------+---------+---------+-----
	 * | Thread0 | Thread0 | Thread0 | ...
	 * | Thread1 | Thread1 | Thread1 | ...
	 * | Thread2 | Thread2 | Thread2 | ...
	 * | ...     | ...     | ...     | ...
	 * To make the function, I started to make smth like that:
	 * | NewerGen | ... | ElderGen |
	 * +----------+-----+----------+
	 * | Person0  | ... | Person0  |
	 * | Person1  | ... | Person1  |
	 * | Person2  | ... | Person2  |
	 * | ...      | ... | ...      |
	 * This solution work, BUT ONLY IF THERE IS A MAX OF 1024 PERSON PER GENS!
	 * (The limit of a GPU, there is not really limit (2^31-1) of blocks...)
	 * But currently, MAX_TH_IN_GEN is set to 2^20.
	 * 2^20 > 2^10
	 * Small error... So the function wouldn't be called.
	 * So I add this if MAX_TH_IN_GEN > 1024:
	 * | AllPersOfNum0 | ... | AllPersOfNum[MAX_TH_IN_GEN] | --> DIM_X
	 * +---------------+-----+-----------------------------+
	 * | PersonOfGen0  | ... | PersonOfGen0                |
	 * | PersonOfGen1  | ... | PersonOfGen1                |
	 * | PersonOfGen2  | ... | PersonOfGen2                |
	 * | ...           | ... | ...                         |
	 * 
	 *        |
	 *        |
	 *        v
	 *      DIM_Y
	 */
	if (tid == 0)
		printf("age=%u\n", ALL[DIM_X][DIM_Y].age);

	//if (ALL[DIM_X][DIM_Y].age == ULLONG_MAX)
	//	printf("ALERT: I was not deleted! tid=%u. ", tid);
	if (DIM_X > *NUM_GENS2 || DIM_Y > *NUM_ELEMS2 ||
		ALL[DIM_X][DIM_Y].age == USHRT_MAX)
		return;

	ALL[DIM_X][DIM_Y].age++;

	if (ALL[DIM_X][DIM_Y].age >=
		min((int)ALL[DIM_X][DIM_Y].max_age[0],
			(int)ALL[DIM_X][DIM_Y].max_age[1]) && 
		ALL[DIM_X][DIM_Y].age != UCHAR_MAX)
	{
		ALL[DIM_X][DIM_Y].age = UCHAR_MAX;
		NUM_ELEMS2[DIM_X]--;
	}
	//else printf("Alive!");
	#undef DIM_X
	#undef DIM_Y
}

/**
 * @brief Check if the last (in the array: older gen) should be deleted.
 * 
 * @param ALL_arr The main array.
 * @return Nothing.
 */
__host__ void deletor(t_o_l (*ALL_arr)[MAX_TH_IN_GEN], nElemTpe *NUM_ELEMS)
{
	for (nElemTpe i = 0; i < MAX_TH_IN_GEN/* && i < *(NUM_ELEMS+NUM_GENS-1)*/; i++)
		if (ALL_arr[NUM_GENS-1][i].age != UCHAR_MAX && ALL_arr[NUM_GENS-1][i].age != 0)
			return;// TODO: optimize with std::find or smth like that
printf("OK");
	//delete[] ALL_arr[j];
	free(ALL_arr[NUM_GENS-1]);
printf("OK");
	NUM_GENS--;
	ALL_arr = (t_o_l(*)[MAX_TH_IN_GEN])malloc(sizeof(t_o_l[MAX_TH_IN_GEN])*NUM_GENS);

	//ALL[j].clear();
	//ALL.erase(ALL.begin() + j);
}

t_o_l **setupHMM(std::vector<std::vector<t_o_l>> &vals, int N)
{
	t_o_l **temp;
	temp = new t_o_l*[N];
	/*for(unsigned i=0; (i < N); i++)
	{ 
		temp[i] = vals[i].data();
	}*/
	temp[0] = vals[0].data();
	return temp;
}

nElemTpe num_elems(nElemTpe *NUM_ELEMS)
{
	auto tmp = NUM_ELEMS;
	nElemTpe s = 0;
	for (int i = 0; i < NUM_GENS; i++, tmp++)
		s += *tmp;
	return s;
}

int main()
{
/**--------------------------------------------------------------------------**/
/**---------------------------DEFINES VERIFICATION PART 2--------------------**/
/**--------------------------------------------------------------------------**/
static_assert(std::is_unsigned_v<nElemTpe>, "nElemTpe must be an unsigned type.");

/**--------------------------------------------------------------------------**/
/**---------------------------MAIN-------------------------------------------**/
/**--------------------------------------------------------------------------**/

	srand(time(NULL));

	nElemTpe *NUM_ELEMS = (nElemTpe*)calloc(10, sizeof(nElemTpe));
	assert(NUM_ELEMS != nullptr);
	NUM_ELEMS[0] = START_COUNT_T;

	//std::vector<std::vector<t_o_l>> ALL(1, std::vector<t_o_l>(NUM_ELEMS[0], t_o_l()));
	// {Last gen, ..., First gen} -> if malloc error Last gen will be conserved
	t_o_l (*ALL_arr)[MAX_TH_IN_GEN] = (t_o_l(*)[MAX_TH_IN_GEN])malloc(sizeof(t_o_l[MAX_TH_IN_GEN])/**NUM_GENS*/);
	assert(ALL_arr != nullptr);
	for (unsigned j = 0; j < NUM_GENS; j++)
		for (nElemTpe i = 0; i < NUM_ELEMS[j]; i++)
			new (ALL_arr[j] + i) t_o_l; // This line calls constructors.
	//t_o_l **ALL_arr = setupHMM(ALL, ALL.size());
	t_o_l **d_ALL;
	size_t ALL_pitch;
	unsigned short *d_NUM_GENS;
	nElemTpe *d_NUM_ELEMS;

	// Validity check
	#ifdef DEBUG
		for (unsigned j = 0; j < NUM_GENS; j++)
			for (nElemTpe i = 0; i < NUM_ELEMS[j] && i < MAX_TH_IN_GEN; i++)
				if (ALL_arr[j][i].validity_check != VALIDITY_CHECK)
				{
					printf("Error: Bad vector init. Element: %u, value: %u, NUM_ELEMS=" nElemTpePrefix ", gen=%u",
							i, ALL_arr[j][i].validity_check, num_elems(NUM_ELEMS), NUM_GENS);
					exit(1);
				}
	#endif

	//std::cout << ALL << std::endl;

	#define COLS sizeof(t_o_l)*MAX_TH_IN_GEN
	
	assert(cudaMallocPitch((void**)&d_ALL,		&ALL_pitch, COLS, NUM_GENS)						== cudaSuccess);
	assert(cudaMalloc((void**)&d_NUM_ELEMS,		sizeof(nElemTpe))								== cudaSuccess);
	assert(cudaMalloc((void**)&d_NUM_GENS,		sizeof(unsigned short))							== cudaSuccess);
	assert(cudaMemcpy2D(d_ALL, ALL_pitch, ALL_arr, COLS, COLS, NUM_GENS,cudaMemcpyHostToDevice)	== cudaSuccess);
	assert(cudaMemcpy(d_NUM_ELEMS,	NUM_ELEMS,	sizeof(nElemTpe),		cudaMemcpyHostToDevice)	== cudaSuccess);
	assert(cudaMemcpy(d_NUM_GENS,	&NUM_GENS,	sizeof(unsigned short),	cudaMemcpyHostToDevice)	== cudaSuccess);

	#undef COLS

	atexit_clean(d_ALL);
	atexit_clean(d_NUM_ELEMS);
	atexit_clean(d_NUM_GENS);

	thread_of_live<<<MAX_TH_IN_GEN,1>>>(d_ALL, d_NUM_ELEMS, d_NUM_GENS);
	auto start = std::chrono::steady_clock::now().time_since_epoch();
	cudaDeviceSynchronize();

	while (NUM_GENS > 0)
	{
		std::cout << num_elems(NUM_ELEMS) << " threads needed. " << std::endl;

		assert(cudaMemcpy(NUM_ELEMS,d_NUM_ELEMS,sizeof(nElemTpe)*NUM_GENS,	cudaMemcpyDeviceToHost)==cudaSuccess);
		assert(cudaMemcpy(&NUM_GENS,d_NUM_GENS,	sizeof(unsigned short),		cudaMemcpyDeviceToHost)==cudaSuccess);
		assert(cudaMemcpy2D(d_ALL,ALL_pitch,ALL_arr,sizeof(t_o_l)*MAX_TH_IN_GEN,sizeof(t_o_l)*MAX_TH_IN_GEN,NUM_GENS,cudaMemcpyHostToDevice)==cudaSuccess);
printf("OK");
		// Check (and delete generation)
		deletor(ALL_arr, NUM_ELEMS);
printf("OK");
		auto end = std::chrono::steady_clock::now().time_since_epoch();
		std::this_thread::sleep_for(YEAR_DURATION_std - (end-start));
		start = std::chrono::steady_clock::now().time_since_epoch();
printf("OK");
		// Validity
		#ifdef DEBUG
			for (unsigned short j = 0; j < NUM_GENS; j++)
				for (nElemTpe i = 0; i < NUM_ELEMS[j]; i++)
					if (ALL_arr[j][i].validity_check != VALIDITY_CHECK)
					{
						printf("Error: Bad vector init. Element: " nElemTpePrefix ", value: %u, NUM_ELEMS=" nElemTpePrefix ", gen=%u",
								i, ALL_arr[j][i].validity_check, num_elems(NUM_ELEMS), NUM_GENS);
						exit(1);
					}
		#endif
printf("OK");
		assert(cudaMallocPitch((void**)&d_ALL,&ALL_pitch, sizeof(t_o_l)*MAX_TH_IN_GEN, NUM_GENS)==cudaSuccess);
		assert(cudaMalloc((void**)d_NUM_ELEMS,		sizeof(nElemTpe)*NUM_GENS)==cudaSuccess);
		assert(cudaMalloc((void**)&d_NUM_GENS,		sizeof(unsigned short))==cudaSuccess);
		assert(cudaMemcpy2D(d_ALL,ALL_pitch,ALL_arr,sizeof(t_o_l)*MAX_TH_IN_GEN,sizeof(t_o_l)*MAX_TH_IN_GEN,NUM_GENS,cudaMemcpyHostToDevice)==cudaSuccess);
		assert(cudaMemcpy(d_NUM_ELEMS,	&NUM_ELEMS,	sizeof(nElemTpe),					cudaMemcpyHostToDevice)==cudaSuccess);
		assert(cudaMemcpy(d_NUM_GENS,	&NUM_GENS,	sizeof(unsigned short),				cudaMemcpyHostToDevice)==cudaSuccess);
printf("OK");
		if (NUM_GENS > 1024)
		{
			printf("Humm... there should be an error. NUM_GENS > 1024.");
			exit(1);
		}
		#if MAX_TH_IN_GEN > 1024
			thread_of_live<<<MAX_TH_IN_GEN,NUM_GENS>>>(d_ALL, d_NUM_ELEMS, d_NUM_GENS);
			// Max blocs: 2^31-1: Max threads in a bloc: 1024; Is a BIG problem for me: true
		#else
			thread_of_live<<<NUM_GEN,MAX_TH_IN_GEN>>>(d_ALL, d_NUM_ELEMS, d_NUM_GENS);
		#endif
		cudaDeviceSynchronize();
	} // TODO: Use 'cuda-memcheck'

	for (int j = 0; j < NUM_GENS; j++)
		for (int i = 0; i < MAX_TH_IN_GEN; i++)
			ALL_arr[j][i].~t_o_l(); // This line calls destructors.

	delete[] ALL_arr;
	cudaFree((void**)&d_ALL);
	cudaFree((void**)&d_NUM_ELEMS);
	cudaFree((void**)&d_NUM_GENS);
	free(NUM_ELEMS);

	printf("Exit OK!");

	return 0;
}
