#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include "HNSW.hpp"
#include "dist.hpp"
using ANN::HNSW;

template<typename T>
point_converter_default<T> to_point;

template<typename U>
void gen_model(commandLine parameter) // intend to be passed in value
{
	char* file_in = parameter.getOptionValue("-in");
	const char* file_out = parameter.getOptionValue("-out");
	const uint32_t cnt_points = parameter.getOptionLongValue("-n", 0);
	const float m_l = parameter.getOptionDoubleValue("-ml", 0.36);
	const uint32_t m = parameter.getOptionIntValue("-m", 40);
	const uint32_t efc = parameter.getOptionIntValue("-efc", 60);
	const float alpha = parameter.getOptionDoubleValue("-alpha", 1);
	const float batch_base = parameter.getOptionDoubleValue("-b", 2);
	const bool do_fixing = !!parameter.getOptionIntValue("-f", 0);

	if(file_in==nullptr || file_out==nullptr)
		throw std::invalid_argument("in/out files are not indicated");
	
	parlay::internal::timer t("HNSW", true);

	using T = typename U::type_elem;
	auto [ps,dim] = load_point(file_in, to_point<T>, cnt_points);
	t.next("Read inFile");

	fputs("Start building HNSW\n", stderr);
	HNSW<U> g(
		ps.begin(), ps.begin()+ps.size(), dim,
		m_l, m, efc, alpha, batch_base, do_fixing
	);
	t.next("Build index");

	g.save(file_out);
	t.next("Write to the file");
}

int main(int argc, char **argv)
{
	for(int i=0; i<argc; ++i)
		printf("%s ", argv[i]);
	putchar('\n');

	commandLine parameter(argc, argv, 
		"-n <numInput> -ml <m_l> -m <m> "
		"-efc <ef_construction> -alpha <alpha> -r <recall@R> [-b <batchBase>]"
		"-in <inFile> -out <modelFile>"
	);

	const char *dist_func = parameter.getOptionValue("-dist");
	auto gen_model_helper = [&](auto type){ // emulate a generic lambda in C++20
		using T = decltype(type);
		if(!strcmp(dist_func,"L2"))
			gen_model<descr_l2<T>>(parameter);
		else if(!strcmp(dist_func,"angular"))
			gen_model<descr_ang<T>>(parameter);
		else throw std::invalid_argument("Unsupported distance type");
	};

	const char* type = parameter.getOptionValue("-type");
	if(!strcmp(type,"uint8"))
		gen_model_helper(uint8_t{});
	else if(!strcmp(type,"int8"))
		gen_model_helper(int8_t{});
	else if(!strcmp(type,"float"))
		gen_model_helper(float{});
	else throw std::invalid_argument("Unsupported element type");
	return 0;
}
