#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <chrono>
#include <stdexcept>
#include "HNSW.hpp"
#include "dist.hpp"
using ANN::HNSW;

parlay::sequence<size_t> per_visited;
parlay::sequence<size_t> per_eval;
parlay::sequence<size_t> per_size_C;

template<typename T>
point_converter_default<T> to_point;

template<typename T>
class gt_converter{
public:
	using type = parlay::sequence<T>;
	template<typename Iter>
	type operator()([[maybe_unused]] uint32_t id, Iter begin, Iter end)
	{
		using type_src = typename std::iterator_traits<Iter>::value_type;
		static_assert(std::is_convertible_v<type_src,T>, "Cannot convert to the target type");

		const uint32_t n = std::distance(begin, end);

		// T *gt = new T[n];
		auto gt = parlay::sequence<T>(n);
		for(uint32_t i=0; i<n; ++i)
			gt[i] = *(begin+i);
		return gt;
	}
};

// Visit all the vectors in the given 2D array of points
// This triggers the page fetching if the vectors are mmap-ed
template<class T>
void visit_point(const T &array, size_t dim0, size_t dim1)
{
	parlay::parallel_for(0, dim0, [&](size_t i){
		const auto &a = array[i];
		[[maybe_unused]] volatile auto elem = a.coord[0];
		for(size_t j=1; j<dim1; ++j)
			elem = a.coord[j];
	});
}

template<class U>
double output_recall(HNSW<U> &g, parlay::internal::timer &t, uint32_t ef, uint32_t k, 
	uint32_t cnt_query, parlay::sequence<typename U::type_point> &q, parlay::sequence<parlay::sequence<uint32_t>> &gt, 
	uint32_t rank_max, float beta, bool warmup, std::optional<float> radius, std::optional<uint32_t> limit_eval)
{
	per_visited.resize(cnt_query);
	per_eval.resize(cnt_query);
	per_size_C.resize(cnt_query);
	//std::vector<std::vector<std::pair<uint32_t,float>>> res(cnt_query);
	parlay::sequence<parlay::sequence<std::pair<uint32_t,float>>> res(cnt_query);
	if(warmup)
	{
		parlay::parallel_for(0, cnt_query, [&](size_t i){
			res[i] = g.search(q[i], k, ef);
		});
	}
	t.next("Doing search");

	parlay::parallel_for(0, cnt_query, [&](size_t i){
		search_control ctrl{};
		ctrl.log_per_stat = i;
		ctrl.beta = beta;
		ctrl.radius = radius;
		ctrl.limit_eval = limit_eval;
		res[i] = g.search(q[i], k, ef, ctrl);
	});
	const double time_query = t.next_time();
	const auto qps = cnt_query/time_query;
	printf("HNSW: Find neighbors: %.4f\n", time_query);

	double ret_val = 0;
	if(radius) // range search
	{
		// -----------------
		float nonzero_correct = 0.0;
		float zero_correct = 0.0;
		uint32_t num_nonzero = 0;
		uint32_t num_zero = 0;
		size_t num_entries = 0;
		size_t num_reported = 0;

		for(uint32_t i=0; i<cnt_query; i++)
		{
			if(gt[i].size()==0)
			{
				num_zero++;
				if(res[i].size()==0)
					zero_correct += 1;
			}
			else
			{
				num_nonzero++;
				size_t num_real_results = gt[i].size();
				size_t num_correctly_reported = res[i].size();
				num_entries += num_real_results;
				num_reported += num_correctly_reported;
				nonzero_correct += float(num_correctly_reported)/num_real_results;
			}
		}
		const float nonzero_recall = nonzero_correct/num_nonzero;
		const float zero_recall = zero_correct/num_zero;
		const float total_recall = (nonzero_correct+zero_correct)/cnt_query;
		const float alt_recall = float(num_reported)/num_entries;

		printf("measure range recall with ef=%u beta=%.4f on %u queries\n", ef, beta, cnt_query);
		printf("query finishes at %ekqps\n", qps/1000);
		printf("#non-zero queries: %u, #zero queries: %u\n", num_nonzero, num_zero);
		printf("non-zero recall: %f, zero recall: %f\n", nonzero_recall, zero_recall);
		printf("total_recall: %f, alt_recall: %f\n", total_recall, alt_recall);
		printf("size of range candidates: %lu\n", parlay::reduce(g.total_range_candidate,parlay::addm<size_t>{}));

		ret_val = nonzero_recall;
	}
	else // k-NN search
	{
		if(rank_max<k)
		{
			fprintf(stderr, "Adjust k from %u to %u\n", k, rank_max);
			k = rank_max;
		}
	//	uint32_t cnt_all_shot = 0;
		std::vector<uint32_t> result(k+1);
		printf("measure recall@%u with ef=%u beta=%.4f on %u queries\n", k, ef, beta, cnt_query);
		for(uint32_t i=0; i<cnt_query; ++i)
		{
			uint32_t cnt_shot = 0;
			for(uint32_t j=0; j<k; ++j)
				if(std::find_if(res[i].begin(),res[i].end(),[&](const std::pair<uint32_t,double> &p){
					return p.first==gt[i][j];}) != res[i].end())
				{
					cnt_shot++;
				}
			result[cnt_shot]++;
		}
		size_t total_shot = 0;
		for(size_t i=0; i<=k; ++i)
		{
			printf("%u ", result[i]);
			total_shot += result[i]*i;
		}
		putchar('\n');
		printf("%.6f at %ekqps\n", float(total_shot)/cnt_query/k, qps/1000);

		ret_val = double(total_shot)/cnt_query/k;
	}
	printf("# visited: %lu\n", parlay::reduce(per_visited,parlay::addm<size_t>{}));
	printf("# eval: %lu\n", parlay::reduce(per_eval,parlay::addm<size_t>{}));
	printf("size of C: %lu\n", parlay::reduce(per_size_C,parlay::addm<size_t>{}));
	if(limit_eval)
		printf("limit the number of evaluated nodes : %u\n", *limit_eval);
	else
		puts("no limit on the number of evaluated nodes");

	parlay::sort_inplace(per_visited);
	parlay::sort_inplace(per_eval);
	parlay::sort_inplace(per_size_C);
	const double tail_ratio[] = {0.9, 0.99, 0.999};
	for(size_t i=0; i<sizeof(tail_ratio)/sizeof(*tail_ratio); ++i)
	{
		const auto r = tail_ratio[i];
		const uint32_t tail_index = r*cnt_query;
		printf("%.4f tail stat (at %u):\n", r, tail_index);

		printf("\t# visited: %lu\n", per_visited[tail_index]);
		printf("\t# eval: %lu\n", per_eval[tail_index]);
		printf("\tsize of C: %lu\n", per_size_C[tail_index]);
	}
	puts("---");
	return ret_val;
}

template<class U>
void output_recall(HNSW<U> &g, commandLine param, parlay::internal::timer &t)
{
	const char* file_query = param.getOptionValue("-q");
	const char* file_groundtruth = param.getOptionValue("-g");
	auto [q,_] = load_point(file_query, to_point<typename U::type_elem>);
	t.next("Read queryFile");
	printf("%s: [%lu,%u]\n", file_query, q.size(), _);

	visit_point(q, q.size(), g.dim);
	t.next("Fetch query vectors");

	auto [gt,rank_max] = load_point(file_groundtruth, gt_converter<uint32_t>{});
	t.next("Read groundTruthFile");
	printf("%s: [%lu,%u]\n", file_groundtruth, gt.size(), rank_max);

	auto parse_array = [](const std::string &s, auto f){
		std::stringstream ss;
		ss << s;
		std::string current;
		std::vector<decltype(f((char*)NULL))> res;
		while(std::getline(ss, current, ','))
			res.push_back(f(current.c_str()));
		std::sort(res.begin(), res.end());
		return res;
	};
	auto beta = parse_array(param.getOptionValue("-beta","1.0"), atof);
	auto cnt_rank_cmp = parse_array(param.getOptionValue("-r"), atoi);
	auto ef = parse_array(param.getOptionValue("-ef"), atoi);
	auto threshold = parse_array(param.getOptionValue("-th"), atof);
	const uint32_t cnt_query = param.getOptionIntValue("-k", q.size());
	const bool enable_warmup = !!param.getOptionIntValue("-w", 1);
	const bool limit_eval = !!param.getOptionIntValue("-le", 0);
	auto radius = [](const char *s) -> std::optional<float>{
			return s? std::optional<float>{atof(s)}: std::optional<float>{};
		}(param.getOptionValue("-rad"));

	auto get_best = [&](uint32_t k, uint32_t ef, std::optional<uint32_t> limit_eval=std::nullopt){
		double best_recall = 0;
		// float best_beta = beta[0];
		for(auto b : beta)
		{
			const double cur_recall = 
				output_recall(g, t, ef, k, cnt_query, q, gt, rank_max, b, enable_warmup, radius, limit_eval);
			if(cur_recall>best_recall)
			{
				best_recall = cur_recall;
				// best_beta = b;
			}
		}
		// return std::make_pair(best_recall, best_beta);
		return best_recall;
	};
	puts("pattern: (k,ef_max,beta)");
	const auto ef_max = *ef.rbegin();
	for(auto k : cnt_rank_cmp)
		get_best(k, ef_max);

	puts("pattern: (k_min,ef,beta)");
	const auto k_min = *cnt_rank_cmp.begin();
	for(auto efq : ef)
		get_best(k_min, efq);

	puts("pattern: (k,threshold)");
	for(auto k : cnt_rank_cmp)
	{
		uint32_t l_last = k;
		for(auto t : threshold)
		{
			printf("searching for k=%u, th=%f\n", k, t);
			const double target = t;
			// const size_t target = t*cnt_query*k;
			uint32_t l=l_last, r_limit=std::max(k*100, ef_max);
			uint32_t r = l;
			bool found = false;
			while(true)
			{
				// auto [best_shot, best_beta] = get_best(k, r);
				if(get_best(k,r)>=target)
				{
					found = true;
					break;
				}
				if(r==r_limit) break;
				r = std::min(r*2, r_limit);
			}
			if(!found) break;
			while(r-l>l*0.05+1) // save work based on an empirical value
			{
				const auto mid = (l+r)/2;
				const auto best_shot = get_best(k,mid);
				if(best_shot>=target)
					r = mid;
				else
					l = mid;
			}
			l_last = l;
		}
	}

	if(limit_eval)
	{
		puts("pattern: (ef_min,k,le,threshold(low numbers))");
		const auto ef_min = *ef.begin();
		for(auto k : cnt_rank_cmp)
		{
			const auto base_shot = get_best(k,ef_min);
			const auto base_eval = parlay::reduce(per_eval,parlay::addm<size_t>{})/cnt_query+1;
			auto base_it = std::lower_bound(threshold.begin(), threshold.end(), base_shot);
			uint32_t l_last = 0; // limit #eval to 0 must keep the recall below the threshold
			for(auto it=threshold.begin(); it!=base_it; ++it)
			{
				uint32_t l=l_last, r=base_eval;
				while(r-l>l*0.05+1)
				{
					const auto mid = (l+r)/2;
					const auto best_shot = get_best(k,ef_min,mid); // limit #eval here
					if(best_shot>=*it)
						r = mid;
					else
						l = mid;
				}
				l_last = l;
			}
		}
	}
}

template<typename U>
void run_test(commandLine parameter) // intend to be pass-by-value manner
{
	const char *file_in = parameter.getOptionValue("-in");
	const uint32_t cnt_points = parameter.getOptionLongValue("-n", 0);
	const float m_l = parameter.getOptionDoubleValue("-ml", 0.36);
	const uint32_t m = parameter.getOptionIntValue("-m", 40);
	const uint32_t efc = parameter.getOptionIntValue("-efc", 60);
	const float alpha = parameter.getOptionDoubleValue("-alpha", 1);
	const float batch_base = parameter.getOptionDoubleValue("-b", 2);
	const bool do_fixing = !!parameter.getOptionIntValue("-f", 0);
	const char *file_out = parameter.getOptionValue("-out");
	
	parlay::internal::timer t("HNSW", true);

	using T = typename U::type_elem;
	auto [ps,dim] = load_point(file_in, to_point<T>, cnt_points);
	t.next("Read inFile");
	printf("%s: [%lu,%u]\n", file_in, ps.size(), dim);

	visit_point(ps, ps.size(), dim);
	t.next("Fetch input vectors");

	fputs("Start building HNSW\n", stderr);
	HNSW<U> g(
		ps.begin(), ps.begin()+ps.size(), dim,
		m_l, m, efc, alpha, batch_base, do_fixing
	);
	t.next("Build index");

	const uint32_t height = g.get_height();
	printf("Highest level: %u\n", height);
	puts("level     #vertices         #degrees  max_degree");
	for(uint32_t i=0; i<=height; ++i)
	{
		const uint32_t level = height-i;
		size_t cnt_vertex = g.cnt_vertex(level);
		size_t cnt_degree = g.cnt_degree(level);
		size_t degree_max = g.get_degree_max(level);
		printf("#%2u: %14lu %16lu %11lu\n", level, cnt_vertex, cnt_degree, degree_max);
	}
	t.next("Count vertices and degrees");

	if(file_out)
	{
		g.save(file_out);
		t.next("Write to the file");
	}

	output_recall(g, parameter, t);
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-type <elemType> -dist <distance> -n <numInput> -ml <m_l> -m <m> "
		"-efc <ef_construction> -alpha <alpha> -f <symmEdge> [-b <batchBase>] "
		"-in <inFile> -out <outFile> -q <queryFile> -g <groundtruthFile> [-k <numQuery>=all] "
		"-ef <ef_query>,... -r <recall@R>,... -th <threshold>,... [-beta <beta>,...] "
		"-le <limit_num_eval> [-w <warmup>] [-rad radius (for range search)]"
	);

	const char *dist_func = parameter.getOptionValue("-dist");
	auto run_test_helper = [&](auto type){ // emulate a generic lambda in C++20
		using T = decltype(type);
		if(!strcmp(dist_func,"L2"))
			run_test<descr_l2<T>>(parameter);
		else if(!strcmp(dist_func,"angular"))
			run_test<descr_ang<T>>(parameter);
		else if(!strcmp(dist_func,"ndot"))
			run_test<descr_ndot<T>>(parameter);
		else throw std::invalid_argument("Unsupported distance type");
	};

	const char* type = parameter.getOptionValue("-type");
	if(!strcmp(type,"uint8"))
		run_test_helper(uint8_t{});
	else if(!strcmp(type,"int8"))
		run_test_helper(int8_t{});
	else if(!strcmp(type,"float"))
		run_test_helper(float{});
	else throw std::invalid_argument("Unsupported element type");
	return 0;
}
