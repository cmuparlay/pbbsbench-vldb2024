#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

extern parlay::sequence<parlay::sequence<std::array<float,5>>> dist_in_search;
extern parlay::sequence<parlay::sequence<std::array<float,5>>> vc_in_search;
// extern parlay::sequence<uint32_t> round_in_search;

#include <optional>

struct search_control{
	bool verbose_output;
	bool skip_search;
	std::optional<uint32_t> log_dist;
	std::optional<uint32_t> log_size;
	std::optional<uint32_t> indicate_ep;
};

#endif // _DEBUG_HPP_
