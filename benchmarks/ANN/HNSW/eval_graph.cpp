#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <string>
#include <map>
// #include <memory>
#include <H5Cpp.h>
#include "HNSW.hpp"
#include "dist.hpp"
#include "debug.hpp"
using ANN::HNSW;

typedef descr_ang<float> descr_fvec;

template<typename T>
point_converter_default<T> to_point;

template<typename T>
class gt_converter{
public:
	using type = T*;
	template<typename Iter>
	type operator()([[maybe_unused]] uint32_t id, Iter begin, Iter end)
	{
		using type_src = typename std::iterator_traits<Iter>::value_type;
		static_assert(std::is_convertible_v<type_src,T>, "Cannot convert to the target type");

		const uint32_t n = std::distance(begin, end);

		T *gt = new T[n];
		for(uint32_t i=0; i<n; ++i)
			gt[i] = *(begin+i);
		return gt;
	}
};

parlay::sequence<parlay::sequence<std::array<float,5>>> dist_in_search;
parlay::sequence<parlay::sequence<std::array<float,5>>> vc_in_search;
// parlay::sequence<uint32_t> round_in_search;

template<typename T>
void store_to_HDF5(const char *file, const T &res, uint32_t bound1, uint32_t bound2)
{
	H5::H5File h5f_out(file, H5F_ACC_TRUNC);
	hsize_t bound_out[2] = {bound1,bound2};
	H5::DataSpace dspace_out(2, bound_out);
	/*
	H5::CompType h5t_dist(sizeof(dist));
	h5t_dist.insertMember("d", offsetof(dist,d), H5::PredType::NATIVE_DOUBLE);
	h5t_dist.insertMember("v", offsetof(dist,v), H5::PredType::NATIVE_UINT32);
	*/
	auto res_d = std::make_unique<float[]>(bound1*bound2);
	auto res_v = std::make_unique<uint32_t[]>(bound1*bound2);
	auto res_dep = std::make_unique<uint32_t[]>(bound1*bound2);
	parlay::parallel_for(0, uint64_t(bound1)*bound2, [&](uint64_t i){
		res_v[i] = std::get<0>(res[i]);
		res_dep[i] = std::get<1>(res[i]);
		res_d[i] = std::get<2>(res[i]);
	});

	H5::DataSet dset_d = h5f_out.createDataSet("distances", H5::PredType::NATIVE_FLOAT, dspace_out);
	dset_d.write(res_d.get(), H5::PredType::NATIVE_FLOAT);
	H5::DataSet dset_v = h5f_out.createDataSet("neighbors", H5::PredType::NATIVE_UINT32, dspace_out);
	dset_v.write(res_v.get(), H5::PredType::NATIVE_UINT32);
	H5::DataSet dset_dep = h5f_out.createDataSet("depth", H5::PredType::NATIVE_UINT32, dspace_out);
	dset_dep.write(res_dep.get(), H5::PredType::NATIVE_UINT32);
}

template<typename T>
void store_to_HDF5(const char *file, const T &res)
{
	const size_t cnt_row = res.size();
	auto size_each = parlay::delayed_seq<size_t>(cnt_row, [&](size_t i){
		return res[i].size();
	});
	const size_t cnt_col = parlay::reduce(size_each, parlay::maxm<size_t>());

	H5::H5File h5f_out(file, H5F_ACC_TRUNC);
	// hsize_t bound_out[2] = {res.size()+1, 128};
	hsize_t bound_out[2] = {cnt_row, cnt_col};
	// hsize_t bound_out_max[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
	H5::DataSpace dspace_out(2, bound_out);

	H5::DSetCreatPropList cparms;
	// hsize_t chunk_dims[2] = {16, 128};
	// cparms.setChunk(2, chunk_dims);
	float fill_val = 0;
	cparms.setFillValue(H5::PredType::NATIVE_FLOAT, &fill_val);

	H5::DataSet dset[5];
	for(uint32_t rank=0; rank<5; ++rank)
	{
		char name[32];
		sprintf(name, "dist_%u", rank*25);
		dset[rank] = h5f_out.createDataSet(name, H5::PredType::NATIVE_FLOAT, dspace_out, cparms);
	}

	for(size_t i=0; i<res.size(); ++i)
	{
		const auto &e = res[i];
		hsize_t dim[2] = {1,e.size()};
		H5::DataSpace dpsace_mem(2, dim);
	/*
		bound_out[1] = e.size();
		dset[0].extend(bound_out);
		dspace_out = dset[0].getSpace();
	*/
		hsize_t offset[2] = {i,0};
		dspace_out.selectHyperslab(H5S_SELECT_SET, dim, offset);

		parlay::sequence<float> res_d(e.size());
		for(uint32_t rank=0; rank<5; ++rank)
		{
			parlay::parallel_for(0, e.size(), [&](size_t j){
				res_d[j] = e[j][rank];
			});
			// dset[rank].extend(bound_out);
			dset[rank].write(res_d.data(), H5::PredType::NATIVE_FLOAT, dpsace_mem, dspace_out);
		}
	}
}

typedef void (*type_func)(HNSW<descr_fvec> &, commandLine, parlay::internal::timer&);

void output_deg(HNSW<descr_fvec> &g, commandLine param, [[maybe_unused]] parlay::internal::timer &t)
{
	if(param.getOption("-?"))
	{
		printf(__func__);
		puts(": -f <outFile> [-l <level>=0]");
		return;
	};
	const char* outfile = param.getOptionValue("-f");
	const uint32_t level = param.getOptionIntValue("-l", 0);
	auto deg = g.get_deg(level);
	FILE *file_deg = fopen(outfile, "w");
	if(!file_deg)
	{
		fputs("fail to create the file\n", stderr);
		exit(2);
	}
	for(auto e : deg)
	{
		fprintf(file_deg, "%u\n", e);
	}
	fclose(file_deg);
}

void output_recall(HNSW<descr_fvec> &g, commandLine param, parlay::internal::timer &t)
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
	const char* file_query = param.getOptionValue("-q");
	const char* file_groundtruth = param.getOptionValue("-g");
	auto [q,_] = load_point(file_query, to_point<float>); (void)_;
	t.next("Read queryFile");

	uint32_t cnt_rank_cmp = param.getOptionIntValue("-r", 1);
	const uint32_t ef = param.getOptionIntValue("-ef", cnt_rank_cmp*50);
	const uint32_t cnt_pts_query = param.getOptionIntValue("-k", q.size());

	std::vector<parlay::sequence<std::pair<uint32_t,float>>> res(cnt_pts_query);
	parlay::parallel_for(0, cnt_pts_query, [&](size_t i){
		res[i] = g.search(q[i], cnt_rank_cmp, ef);
	});
	t.next("Find neighbors");

	auto [gt,rank_max] = load_point(file_groundtruth, gt_converter<uint32_t>{});

	if(rank_max<cnt_rank_cmp)
		cnt_rank_cmp = rank_max;
	uint32_t cnt_all_shot = 0;
	printf("measure recall@%u\n", cnt_rank_cmp);
	for(uint32_t i=0; i<cnt_pts_query; ++i)
	{
		uint32_t cnt_shot = 0;
		for(uint32_t j=0; j<cnt_rank_cmp; ++j)
			if(std::find_if(res[i].begin(),res[i].end(),[&](const std::pair<uint32_t,double> &p){
				return p.first==gt[i][j];}) != res[i].end())
			{
				cnt_shot++;
			}
		printf("#%u:\t%u (%.2f)[%lu]", i, cnt_shot, float(cnt_shot)/cnt_rank_cmp, res[i].size());
		if(cnt_shot==cnt_rank_cmp)
		{
			cnt_all_shot++;
		}
		putchar('\n');
	}
	printf("#all shot: %u (%.2f)\n", cnt_all_shot, float(cnt_all_shot)/cnt_pts_query);
	printf("# visited: %lu\n", parlay::reduce(g.total_visited,parlay::addm<size_t>{}));
	printf("# eval: %lu\n", parlay::reduce(g.total_eval,parlay::addm<size_t>{}));
	printf("size of C: %lu\n", parlay::reduce(g.total_size_C,parlay::addm<size_t>{}));
}

void query(HNSW<descr_fvec> &g, commandLine param, parlay::internal::timer &t)
{
	if(param.getOption("-?"))
	{
		printf(__func__);
		puts(
			"-q <queryFile> -o <outputFile>"
			"-ef <ef_query> [-r <recall@R>=1] [-k <numQuery>=all] [-d <distDump>]"
		);
		return;
	};
	char* file_query = param.getOptionValue("-q");
	auto [q,_] = load_point(file_query, to_point<float>); (void)_;
	t.next("Read queryFile");

	uint32_t cnt_rank_cmp = param.getOptionIntValue("-r", 1);
	const uint32_t ef = param.getOptionIntValue("-ef", cnt_rank_cmp*50);
	const uint32_t cnt_pts_query = param.getOptionIntValue("-k", q.size());
	const char* dump_dist = param.getOptionValue("-d");

	if(dump_dist)
	{
		dist_in_search.resize(cnt_pts_query);
		vc_in_search.resize(cnt_pts_query);
		// round_in_search.resize(cnt_pts_query);
	}

	std::vector<std::tuple<uint32_t,uint32_t,float>> res(cnt_pts_query*cnt_rank_cmp);
	search_control ctrl{};
	parlay::parallel_for(0, cnt_pts_query, [&](size_t i){
		if(dump_dist)
			ctrl.log_dist = ctrl.log_size = i;
		const auto t = g.search_ex(q[i], cnt_rank_cmp, ef, ctrl);
		for(uint32_t j=0; j<cnt_rank_cmp; ++j)
			res[i*cnt_rank_cmp+j] = t[j];
	});
	t.next("Find neighbors");

	const char* file_out = param.getOptionValue("-o");
	store_to_HDF5(file_out, res, cnt_pts_query, cnt_rank_cmp);
	t.next("Write to output file");

	if(dump_dist)
	{
		store_to_HDF5(dump_dist, dist_in_search);
		char name_vc[64];
		sprintf(name_vc, "%s.vc", dump_dist);
		store_to_HDF5(name_vc, vc_in_search);
		t.next("Write the dump file");
		for(size_t i=0; i<cnt_pts_query; ++i)
			fprintf(stderr, "%ld\n", dist_in_search[i].size());
	}
}

void output_neighbor(HNSW<descr_fvec> &g, commandLine param, [[maybe_unused]] parlay::internal::timer &t)
{
	if(param.getOption("-?"))
	{
		printf(__func__);
		puts("[-q <queryFile>] [<ef_query> <recall> <begin> <end> <stripe>]...");
		return;
	};
	char* file_query = param.getOptionValue("-q");
	auto [q,_] = load_point(file_query, to_point<float>); (void)_;

	puts("Please input <ef_query> <recall> <begin> <end> <stripe> in order");
	while(true)
	{
		uint32_t ef, recall, begin, end, stripe;
		scanf("%u%u%u%u%u", &ef, &recall, &begin, &end, &stripe);
		for(uint32_t i=begin; i<end; i+=stripe)
		{
			search_control ctrl{};
			ctrl.verbose_output = true;
			auto res = g.search_ex(q[i], recall, ef, ctrl);
			printf("Neighbors of %u\n", i);
			for(auto it=res.crbegin(); it!=res.crend(); ++it)
			{
				const auto [id,dep,dis] = *it;
				printf("  [%u]\t%u\t%.6f\n", dep, id, dis);
			}
			putchar('\n');
		}
	}
}

uint32_t cnt_hit(const parlay::sequence<std::tuple<uint32_t,uint32_t,float>> &res, const uint32_t *gt, uint32_t recall)
{
	uint32_t cnt = 0;
	for(uint32_t j=0; j<recall; ++j)
		if(std::find_if(res.begin(),res.end(),[&](const std::tuple<uint32_t,uint32_t,float> &p){
			return std::get<0>(p)==gt[j];}) != res.end())
		{
			cnt++;
		}
	return cnt;
}

void recall_single(HNSW<descr_fvec> &g, commandLine param, [[maybe_unused]] parlay::internal::timer &t)
{
	if(param.getOption("-?"))
	{
		printf(__func__);
		puts(" -q <queryFile> -g <groundtruthFile> [<ef_query> <recall> <query_index>] <ep>...");
		return;
	};
	char* file_query = param.getOptionValue("-q");
	char* file_groundtruth = param.getOptionValue("-g");
	auto [q,_] = load_point(file_query, to_point<float>); (void)_;
	auto [gt,rank_max] = load_point(file_groundtruth, gt_converter<uint32_t>{});

	puts("Please input <ef_query> <recall> <query_index> <ep> in order");
	while(true)
	{
		uint32_t ef, recall, qidx, enum_ep;
		scanf("%u%u%u%u", &ef, &recall, &qidx, &enum_ep);
		if(recall>rank_max)
		{
			fputs("recall is larger than the groundtruth rank", stderr);
			continue;
		}

		search_control ctrl{};
		ctrl.verbose_output = true;
		if(enum_ep)
		{
			parlay::sequence<uint32_t> quality(g.n);
			parlay::parallel_for(0, g.n, [&](size_t i){
				ctrl.indicate_ep = i;
				auto res = g.search_ex(q[qidx], recall, ef, ctrl);
				quality[i] = cnt_hit(res, gt[qidx], recall);
			});
			uint32_t ep = parlay::max_element(quality)-quality.begin();
			ctrl.indicate_ep = ep;
			printf("Choose the best ep of %u\n", ep);
		}

		auto res = g.search_ex(q[qidx], recall, ef, ctrl);
		printf("Neighbors of %u\n", qidx);
		for(auto it=res.cbegin(); it!=res.cend(); ++it)
		{
			const auto [id,dep,dis] = *it;
			printf("  [%u]\t%u\t%.6f\n", dep, id, dis);
		}
		putchar('\n');
		uint32_t hit = cnt_hit(res, gt[qidx], recall);
		printf("recall: %u/%u = %.2f%%\n", hit, recall, float(hit)/recall);
	}
}

void count_number(HNSW<descr_fvec> &g, commandLine param, [[maybe_unused]] parlay::internal::timer &t)
{
	if(param.getOption("-?"))
	{
		printf(__func__);
		return;
	};
	std::map<uint32_t,uint32_t> cnt;
	for(const auto &e : g.node_pool)
		cnt[e.level]++;
	
	uint32_t sum = 0;
	for(int i=cnt.rbegin()->first; i>=0; --i)
		printf("#nodes in lev. %d: %u (%u)\n", i, sum+=cnt[i], cnt[i]);
}

void symmetrize(HNSW<descr_fvec> &g, commandLine param, [[maybe_unused]] parlay::internal::timer &t)
{
	if(param.getOption("-?"))
	{
		puts("-type <symmetrization> -out <modelFile>");
		return;
	};
	
	const char* type_symm = param.getOptionValue("-type");
	if(strcmp(type_symm,"heur")==0)
	{
		g.fix_edge();
	}
	else puts("Unrecognized symmetrization");

	const char* file_out = param.getOptionValue("-out");
	g.save(file_out);
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine param(argc, argv, 
		"-in <inFile> -mod <modelFile> -func <function>"
	);
	char* file_in = param.getOptionValue("-in");
	const char* file_model = param.getOptionValue("-mod");
	const char* func = param.getOptionValue("-func");

	parlay::internal::timer t("HNSW", true);
	auto [ps,_] = load_point(file_in, to_point<float>); (void)_;
	t.next("Read inFile");

	fputs("Start building HNSW\n", stderr);
	HNSW<descr_fvec> g(file_model, [&](uint32_t i){
		return ps[i];
	});
	t.next("Build index");

	std::map<std::string,type_func> list_func;
	list_func["deg"] = output_deg;
	list_func["recall"] = output_recall;
	list_func["neighbor"] = output_neighbor;
	list_func["count"] = count_number;
	list_func["query"] = query;
	list_func["symm"] = symmetrize;
	list_func["single"] = recall_single;

	auto it_func = list_func.find(func);
	if(it_func==list_func.end())
	{
		fprintf(stderr, "Cannot find function '%s'\n", func);
		return 1;
	}

	it_func->second(g, param, t);

	return 0;
}
