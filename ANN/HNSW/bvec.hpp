#ifndef __TYPE_POINT_HPP__
#define __TYPE_POINT_HPP__

#include <cstdint>
#include <iterator>
#include "benchUtils.h"
//#include "h5_ops.hpp"

struct bvec
{
	uint32_t id;
	const uint8_t *coord;
};

std::pair<parlay::sequence<bvec>,uint32_t> ps_from_SIFT(const char *file, size_t max_num=0)
{
	static auto to_bvec = [](size_t id, auto begin, auto end){
		typedef typename std::iterator_traits<decltype(begin)>::value_type type_elem;
		if constexpr(std::is_same_v<decltype(begin),ptr_mapped<type_elem,ptr_mapped_src::DISK>>)
		{
			const auto *begin_raw=begin.get(), *end_raw=end.get();
			const auto n = std::distance(begin_raw, end_raw);

			// auto coord = std::make_unique<type_elem[]>(n);
			/*
			type_elem *coord = new type_elem[n];
			parlay::parallel_for(0, n, [&](size_t i){
				coord[i] = *(begin_raw+i);
			});
			*/
			const type_elem *coord = begin_raw;

			bvec point;
			point.id = id;
			point.coord = std::move(coord);
			return point;
		}
	};

	return parse_vecs<uint8_t>(file, to_bvec, max_num);
}

std::pair<parlay::sequence<bvec>,uint32_t> ps_from_HDF5(const char *file, const char *dir)
{
	//auto [buffer_ptr,bound] = read_array_from_HDF5<uint8_t>(file, dir);
	const uint32_t bound[] = {0,0};
	const auto dim = bound[1];
	//auto *buffer = buffer_ptr.release();
	auto *buffer = (uint8_t*)nullptr;

	parlay::sequence<bvec> ps(bound[0]);
	parlay::parallel_for(0, bound[0], [&](uint32_t i){
		ps[i] = bvec{i, &buffer[i*dim]};
	});
	return {ps,dim};
}
#endif // __TYPE_POINT_HPP_
