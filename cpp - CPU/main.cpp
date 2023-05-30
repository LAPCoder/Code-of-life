#include <iostream>
#include <cstdio>
#include <chrono>

#define YEAR_DURATION std::chrono::milliseconds{100}

#include "./funcs.hpp"

std::list<std::thread*> exe_threads;

/*
auto operator[](auto i, unsigned u)
{
	auto i_front = i.begin();
	return std::advance(i_front, u);
}// TODO: do that
*/

//typedef unsigned long long ull;

int main()
{
	std::cout << "\033[J\033[3J";
	std::shared_ptr<std::vector<std::shared_ptr<coupling_cv>>> COUPLING_LIST = 
		std::make_shared<std::vector<std::shared_ptr<coupling_cv>>>();
	std::shared_ptr<std::list<thread_of_life*>> threads = 
		std::make_shared<std::list<thread_of_life*>>();
	
	for (int i = 0; i < 5; i++)
	{
		auto t = new thread_of_life(COUPLING_LIST, threads, threads->size()+0);
		// +0 to create a copy of the object
		threads->push_back(t);
		std::thread et(&t->run, t);
		//exe_threads.push_back(&et);
		et.detach();
		// TODO: optimize with std::advance(..., 1)
	}
	while (COUPLING_LIST.use_count() == 0);
	//exe_threads.clear();
	while (COUPLING_LIST.use_count() > 1)
	// 1 because the main thread (this) use it
	{
		printf("\033[H\033[J\033[3J%s\n",
			std::to_string(COUPLING_LIST.use_count()).c_str());
		auto tmp = *COUPLING_LIST;
		for (auto c : tmp)
    		printf("- %u %d %d\n",	c->age, c->asked, c->coupled);
		std::this_thread::sleep_for(YEAR_DURATION);
	}

	return 0;
}
