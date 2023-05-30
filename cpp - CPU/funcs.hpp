#ifndef FUNCS_HPP
#define FUNCS_HPP
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>
#include <memory>
#include <vector>
#include <thread>
#include <list>
#include <iterator>

struct coupling_cv
{
	std::thread::id id;
	unsigned short age;
	std::thread::id coupled_with;
	bool coupled = false;
	bool asked = false;
	std::size_t pos_in_thread;
};

class thread_of_life
{
	public:
		typedef unsigned short caracts_us; // us = unsigned short
		typedef const bool caracts_cb; // cb = const_bool
		typedef const unsigned short caracts_cus; // cus = const unsigned short
		typedef const unsigned short caracts_cus2[2];
		// cus2 = const unsigned short[2]
		
		void init();

		// Default
		explicit thread_of_life(
				std::shared_ptr<std::vector<std::shared_ptr<coupling_cv>>> CL,
				std::shared_ptr<std::list<thread_of_life*>> threads,
				std::size_t pos_in_thread_list):
			max_age{(caracts_us)(rand() % 140 + 10),
					(caracts_us)(rand() % 140 + 10)},
			COUPLING_LIST(CL),
			THREADS(threads),
			pos_in_thread(pos_in_thread_list)
			{init();}
		// All
		explicit thread_of_life(
				caracts_us age1,
				caracts_us age2,
				std::shared_ptr<std::vector<std::shared_ptr<coupling_cv>>> CL,
				std::shared_ptr<std::list<thread_of_life*>> threads,
				std::size_t pos_in_thread_list):
			max_age{age1, age2},
			COUPLING_LIST(CL),
			THREADS(threads),
			pos_in_thread(pos_in_thread_list)
			{init();}
		~thread_of_life()
		{
			//std::stringstream ss; ss << "Killed " << std::this_thread::get_id()
			//	<< " at " << std::to_string(cv->age) << " years.\n";
			//std::cout << ss.str();
			COUPLING_LIST->erase(std::find(COUPLING_LIST->begin(),
										   COUPLING_LIST->end(), cv));
			//delete this;
		}

		void set_hp(caracts_us hp1, caracts_us hp2)
		{hp = std::min(hp1, hp2);}

		inline void run();
		inline bool ask_couple(std::shared_ptr<coupling_cv>);

		caracts_us hp; // max = 100
		caracts_cus2 max_age;
		const std::shared_ptr<std::vector<std::shared_ptr<coupling_cv>>>
			COUPLING_LIST;
		std::shared_ptr<coupling_cv> cv = std::make_shared<coupling_cv>();
		const std::shared_ptr<std::list<thread_of_life*>> THREADS;
		const std::size_t pos_in_thread;
		std::size_t cpl_pos_in_thread;
};

void thread_of_life::init()
{
	cv->id = std::this_thread::get_id();
	cv->age = 0;
	cv->pos_in_thread = pos_in_thread;
}

void thread_of_life::run()
{
	//__THREADS_ALIVE--;
	cv->age++;
	//std::cout<<"I am alive!\n";
	if (cv->age >= std::min(max_age[0], max_age[1]))
		delete this;//this->~thread_of_life();
	else
	{
		if (cv->age == 18)
		{
			COUPLING_LIST->push_back(cv);
		}
		// Do childs
		if (cv->coupled && rand()%40 == 1)
		{
			auto vi = THREADS->begin();
			std::advance(vi, cpl_pos_in_thread);
			auto t = new thread_of_life(max_age[0], (*vi)->max_age[0], 
				COUPLING_LIST, THREADS, THREADS->size());
			THREADS->push_back(t);
			std::thread et(&t->run, t);
			et.detach();
		}
		if (cv->age >= 18 && !cv->coupled)
		{
			if (COUPLING_LIST->size() > 1)
			{
				unsigned long r = rand() % (COUPLING_LIST->size()-1);
				try {
					auto tmp = *COUPLING_LIST;
					if (!tmp.at(r)->coupled &&
						!(tmp.at(r)->id == std::this_thread::get_id()))
					{
						auto vi = THREADS->begin();
						cv->asked = true;
						std::advance(vi,tmp.at(r)->pos_in_thread);
						if (!(*vi)->ask_couple(cv))
							cv->asked = false;
						else
							cpl_pos_in_thread = (*vi)->pos_in_thread;

					}
				} catch (std::out_of_range&) {}
			}
		}
		//__THREADS_ALIVE++;
		auto end = std::chrono::system_clock::now() + YEAR_DURATION;
		do {
		} while (std::chrono::system_clock::now() < end);
		run();
	}
}

bool thread_of_life::ask_couple(std::shared_ptr<coupling_cv> ask_cv)
{
	if (!cv->asked && !cv->coupled)
	{// TODO: diff methods: can(not) accept if asked
		cv->coupled = true;
		ask_cv->coupled = true;
		ask_cv->coupled_with = std::this_thread::get_id();
		cv->coupled_with = ask_cv->id;
		return true; // TODO: coupling chances
	}
	else return false; // Decline
}

#endif
