#ifndef __DIST_L2_HPP__
#define __DIST_L2_HPP__

#include <type_traints>
#include "type_point.hpp"

template<typename T>
class descr_l2
{
	using promoted_type = std::conditional_t<std::is_integral_v<T>&&sizeof(T)<=4,
		std::conditional_t<sizeof(T)==4, int64_t, int32_t>,
		float
	>;
public:
	typedef point<T> type_point;
	static float distance(const type_point &u, const type_point &v, uint32_t dim)
	{
		const auto *uc=u.coord, *vc=v.coord;
		promoted_type sum = 0;
		for(uint32_t i=0; i<dim; ++i)
		{
			const auto d = promoted_type(uc[i])-vc[i];
			sum += d*d;
		}
		return sum;
	}

	static auto get_id(const type_point &u)
	{
		return u.id;
	}
};

#endif // _DIST_L2_HPP_
