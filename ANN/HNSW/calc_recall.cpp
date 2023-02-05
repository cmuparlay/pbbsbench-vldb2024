#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <string>
#include <map>
#include <chrono>
// #include <memory>
//#include <H5Cpp.h>
#include "HNSW.hpp"
//#include "dist_ang.hpp"
#include "dist_l2.hpp"
using ANN::HNSW;

// it will change the file string
/*
auto load_fvec(char *file)
{
	char *spec_input = std::strchr(file, ':');
	if(spec_input==nullptr)
	{
		fputs("Unrecognized file spec",stderr);
		return std::make_pair(parlay::sequence<fvec>(),uint32_t(0));
	}
	
	*(spec_input++) = '\0';
	parlay::sequence<fvec> ps;
	uint32_t dim = 0;
	if(spec_input[0]=='/')
		std::tie(ps,dim) = ps_from_HDF5(file, spec_input);
	else if(!std::strcmp(spec_input,"fvec"))
		std::tie(ps,dim) = ps_from_SIFT(file);
	else fputs("Unsupported file spec",stderr);

	return std::make_pair(ps,dim);
}
*/
auto load_bvec(char *file, size_t max_num=0)
{
	char *spec_input = std::strchr(file, ':');
	if(spec_input==nullptr)
	{
		fputs("Unrecognized file spec",stderr);
		return std::make_pair(parlay::sequence<bvec>(),uint32_t(0));
	}
	
	*(spec_input++) = '\0';
	parlay::sequence<bvec> ps;
	uint32_t dim = 0;
	if(spec_input[0]=='/')
		std::tie(ps,dim) = ps_from_HDF5(file, spec_input);
	else if(!std::strcmp(spec_input,"bvec"))
		std::tie(ps,dim) = ps_from_SIFT(file, max_num);
	else fputs("Unsupported file spec",stderr);

	return std::make_pair(std::move(ps),dim);
}

// it will change the file string
auto load_ivec(char *file)
{
	char *spec_input = std::strchr(file, ':');
	if(spec_input==nullptr)
	{
		fputs("Unrecognized file spec",stderr);
		return std::make_pair(parlay::sequence<uint32_t*>(),uint32_t(0));
	}
	
	*(spec_input++) = '\0';
	parlay::sequence<uint32_t*> vec;
	uint32_t bound1 = 0;
	if(spec_input[0]=='/')
	{
		/*
		auto [buffer_ptr,bound] = read_array_from_HDF5<uint32_t>(file, spec_input);
		bound1 = bound[1];
		auto *buffer = buffer_ptr.release();

		vec.resize(bound[0]);
		parlay::parallel_for(0, bound[0], [&](uint32_t i){
			vec[i] = &buffer[i*bound1];
		});
		*/
	}
	else if(!std::strcmp(spec_input,"ivec"))
		std::tie(vec,bound1) = parse_vecs<uint32_t>(file, [](size_t, auto begin, auto end){
			typedef typename std::iterator_traits<decltype(begin)>::value_type type_elem;
			if constexpr(std::is_same_v<decltype(begin),ptr_mapped<type_elem,ptr_mapped_src::DISK>>)
			{
				const auto *begin_raw=begin.get(), *end_raw=end.get();
				const auto n = std::distance(begin_raw, end_raw);

				type_elem *id = new type_elem[n];
				parlay::parallel_for(0, n, [&](size_t i){
					id[i] = static_cast<type_elem>(*(begin_raw+i));
				});
				return id;
			}
		});
	else fputs("Unsupported file spec",stderr);

	return std::make_pair(vec,bound1);
}

void output_recall(HNSW<descr_bvec> &g, parlay::internal::timer &t, uint32_t ef, uint32_t recall, uint32_t cnt_query, parlay::sequence<bvec> &q, parlay::sequence<uint32_t*> &gt, uint32_t rank_max)
{
	g.total_visited = 0;
	g.total_eval = 0;
	g.total_size_C = 0;
	//std::vector<std::vector<std::pair<uint32_t,float>>> res(cnt_query);
	parlay::sequence<parlay::sequence<std::pair<uint32_t,float>>> res(cnt_query);
	parlay::parallel_for(0, cnt_query, [&](size_t i){
		res[i] = g.search(q[i], recall, ef);
	});
	t.next("Doing search");
	//auto t1 = std::chrono::high_resolution_clock::now();
	parlay::parallel_for(0, cnt_query, [&](size_t i){
		// flag_query
		search_control ctrl{};
		res[i] = g.search(q[i], recall, ef, ctrl);
	});
	//auto t2 = std::chrono::high_resolution_clock::now();
	double time_query = t.next_time();
	//printf("time diff: %.8f\n", time_query);
	// auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1);
	//std::chrono::duration<double, std::milli> diff = t2-t1;
	//printf("time diff (hi): %.4f\n", diff.count());
	// t.report(time_query, "Find neighbors");
	printf("HNSW: Find neighbors: %.4f\n", time_query);

	if(rank_max<recall)
		recall = rank_max;
//	uint32_t cnt_all_shot = 0;
	std::vector<uint32_t> result(recall+1);
	printf("measure recall@%u with ef=%u on %u queries\n", recall, ef, cnt_query);
	for(uint32_t i=0; i<cnt_query; ++i)
	{
		uint32_t cnt_shot = 0;
		for(uint32_t j=0; j<recall; ++j)
			if(std::find_if(res[i].begin(),res[i].end(),[&](const std::pair<uint32_t,double> &p){
				return p.first==gt[i][j];}) != res[i].end())
			{
				cnt_shot++;
			}
		/*
		printf("#%u:\t%u (%.2f)[%lu]", i, cnt_shot, float(cnt_shot)/recall, res[i].size());
		if(cnt_shot==recall)
		{
			cnt_all_shot++;
		}
		putchar('\n');
		*/
		result[cnt_shot]++;
	}
	// printf("#all shot: %u (%.2f)\n", cnt_all_shot, float(cnt_all_shot)/cnt_query);
	uint32_t cnt_shot = 0;
	for(uint32_t i=0; i<=recall; ++i)
	{
		printf("%u ", result[i]);
		cnt_shot += result[i]*i;
	}
	putchar('\n');
	printf("%.6f at %ekqps\n", float(cnt_shot)/cnt_query/recall, cnt_query/time_query/1000);
	printf("# visited: %lu\n", g.total_visited.load());
	printf("# eval: %lu\n", g.total_eval.load());
	printf("size of C: %lu\n", g.total_size_C.load());
	puts("---");
}

void output_recall(HNSW<descr_bvec> &g, commandLine param, parlay::internal::timer &t)
{
	if(param.getOption("-?"))
	{
		printf(__func__);
		puts(
			"[-q <queryFile>] [-g <groundtruthFile>]"
			"-ef <ef_query> [-r <recall@R>=1] [-k <numQuery>=all]"
		);
		return;
	};
	char* file_query = param.getOptionValue("-q");
	char* file_groundtruth = param.getOptionValue("-g");
	auto [q,_] = load_bvec(file_query);
	t.next("Read queryFile");

	uint32_t cnt_rank_cmp = param.getOptionIntValue("-r", 1);
//	const uint32_t ef = param.getOptionIntValue("-ef", cnt_rank_cmp*50);
	const uint32_t cnt_pts_query = param.getOptionIntValue("-k", q.size());

	auto [gt,rank_max] = load_ivec(file_groundtruth);
	for(uint32_t scale=1; scale<60; scale+=2)
		output_recall(g, t, scale*cnt_rank_cmp, cnt_rank_cmp, cnt_pts_query, q, gt, rank_max);
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-n <numInput> -ml <m_l> -m <m> "
		"-efc <ef_construction> -alpha <alpha> -r <recall@R> [-b <batchBase>]"
		"-in <inFile> ..."
	);
	char* file_in = parameter.getOptionValue("-in");
	const uint32_t cnt_points = parameter.getOptionLongValue("-n", 0);
	const float m_l = parameter.getOptionDoubleValue("-ml", 0.36);
	const uint32_t m = parameter.getOptionIntValue("-m", 40);
	const uint32_t efc = parameter.getOptionIntValue("-efc", 60);
	const float alpha = parameter.getOptionDoubleValue("-alpha", 1);
	const float batch_base = parameter.getOptionDoubleValue("-b", 2);
	const bool do_fixing = !!parameter.getOptionIntValue("-f", 0);
	flag_query = parameter.getOptionIntValue("-flag", 0);

	if(file_in==nullptr)
		return fputs("in file is not indicated\n",stderr), 1;
	
	char *spec_input = std::strchr(file_in, ':');
	if(spec_input==nullptr)
		return fputs("Unrecognized file spec",stderr), 2;
	
	parlay::internal::timer t("HNSW", true);

	auto [ps,dim] = load_bvec(file_in, cnt_points);
	t.next("Read inFile");

	fputs("Start building HNSW\n", stderr);
	HNSW<descr_bvec> g(
		ps.begin(), ps.begin()+cnt_points, dim,
		m_l, m, efc, alpha, batch_base, do_fixing
	);
	t.next("Build index");

	size_t cnt_degree = g.cnt_degree();
	printf("total degree: %lu\n", cnt_degree);
	t.next("Count degrees");

	output_recall(g, parameter, t);

	return 0;
}
