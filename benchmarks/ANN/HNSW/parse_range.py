import sys
import re
import functools
import csv

p_cmd = re.compile(r"\./calc_recall -n ([0-9]*) .*?-m ([0-9]*) .*?-efc ([0-9]*)")
p_algo = re.compile(r"Start building (.*)")
p_buildtime = re.compile(r"Build index: (.*)")
p_deg0 = re.compile(r"# 0: *([0-9]*) *([0-9]*) *([0-9]*)")
p_read_gt = re.compile(r"Read groundTruthFile")
p_qparam = re.compile(r"measure range recall with ef=([0-9]*) beta=([0-9\.]*) on ([0-9]*) queries")
p_qps = re.compile(r"query finishes at (.*)kqps")
p_nz_queries = re.compile(r"#non-zero queries: ([0-9]*),")
p_recall = re.compile(r"non-zero recall: (.*),")
p_alt_recall = re.compile(r"alt_recall: (.*)")
p_stat99 = re.compile(r"0\.9900 tail stat")
p_cmp_total = re.compile(r"^# visited: ([0-9\.]*)")
p_visit_total = re.compile(r"^# eval: ([0-9\.]*)")
p_cmp_tailing = re.compile(r"\t# visited: ([0-9\.]*)")
p_visit_tailing = re.compile(r"\t# eval: ([0-9\.]*)")
p_stat_tailing = re.compile(r"tail stat")
p_endq = re.compile(r"---")

file_in = sys.argv[1]
file_out = sys.argv[2] if len(sys.argv)>2 else None

preamble = {}
building = {}
query = []
q = {}
state = "PREAMBLE"

f = open(file_in)
for l in f.readlines():
	if state=="PREAMBLE":
		res = re.search(p_cmd, l)
		if res is not None: # commandline
			n = int(res.group(1))
			preamble["scale"] = n
			preamble["scale_M"] = n/1000000
			building["m"] = int(res.group(2))
			building["efc"] = int(res.group(3))
			continue
		res = re.search(p_algo, l)
		if res is not None: # algorithm name
			preamble["algo"] = res.group(1)
			continue
		res = re.search(p_buildtime, l)
		if res is not None: # building time
			building["time"] = float(res.group(1))
			state = "BUILD"
			continue

	if state=="BUILD":
		res = re.search(p_deg0, l)
		if res is not None: # degree info
			building["avg_deg"] = float(res.group(2))/int(res.group(1))
			building["max_deg"] = int(res.group(3))
			continue
		res = re.search(p_read_gt, l)
		if res is not None: # load ground truth
			state = "QUERY"
			continue

	if state=="QUERY":
		res = re.search(p_qparam, l)
		if res is not None: # query param
			q["ef"] = int(res.group(1))
			q["beta"] = float(res.group(2))
			q["cnt"] = int(res.group(3))
			continue
		res = re.search(p_qps, l)
		if res is not None: # QPS
			q["QPS"] = float(res.group(1))*1000
			continue
		res = re.search(p_nz_queries, l)
		if res is not None: # #non-zero queries
			q["num_nz"] = int(res.group(1))
			continue
		res = re.search(p_recall, l)
		if res is not None: # recall
			q["recall_nz"] = float(res.group(1))
			q["recall"] = q["recall_nz"]
			continue
		res = re.search(p_alt_recall, l)
		if res is not None: # alt recall
			q["recall_alt"] = float(res.group(1))
			continue
		res = re.search(p_cmp_total, l)
		if res is not None: # the total number of comparisons
			q["avg_cmp"] = float(res.group(1))/q["cnt"]
			continue
		res = re.search(p_visit_total, l)
		if res is not None: # the total number of visits
			q["avg_visit"] = float(res.group(1))/q["cnt"]
			continue
		res = re.search(p_stat99, l)
		if res is not None: # 0.99 tailing statistics
			state = "TAILING_STAT"
			continue
		res = re.match(p_endq, l)
		if res is not None: # at the end of current query
			query.append(q)
			q = {}

	if state=="TAILING_STAT":
		res = re.search(p_cmp_tailing, l)
		if res is not None: # the tailing number of comparisons
			q[".99_cmp"] = int(res.group(1))
			continue
		res = re.search(p_visit_tailing, l)
		if res is not None: # the tailing number of visits
			q[".99_visit"] = int(res.group(1))
			continue
		res = re.search(p_stat_tailing, l)
		if res is not None: # 0.99 tailing statistics
			state = "QUERY"
			continue

f.close()

print(preamble)
print(building)
# print(query[0])

query.sort(key=lambda q: q["recall"])
# print(query[0])

print("ef\trecall\t\tQPS\t\tavg_cmp")
bucket = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.83, 0.85, 0.87, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999, 1]
q_in_bucket = []
for j in range(len(bucket)-1):
	b = bucket[j]
	l = 0
	r = len(query)
	while l+1<r:
		mid = (l+r)//2
		if query[mid]["recall"]<b:
			l = mid
		else:
			r = mid

	d = (bucket[j+1]-b)*0.2
	R = b + d
	L = b - d/4

	candidate = []
	for q in query:
		recall = q["recall"]
		if L<=recall:
			candidate.append(q)

	"""
	size_limit = 5
	for i in range(l,-1,-1):
		q = query[i]
		if q["recall"]<L:
			break
		candidate.append(q)
		if len(candidate)>size_limit/2:
			break

	for i in range(r,len(q)):
		q = query[i]
		if q["recall"]>R:
			break
		candidate.append(q)
		if len(candidate)>size_limit:
			break
	"""

	print("====== [target recall: %f] ======"%b)
	if len(candidate)>0:
		q = functools.reduce(lambda x,y: x if x["QPS"]>y["QPS"] else y, candidate)
		print("%d\t%f\t%f\t%d"%(q["ef"],q["recall"],q["QPS"],int(q["avg_cmp"])))
		q_in_bucket.append((q,b))
		# recall_in_bucket.append(q["recall"])
		# QPS_in_bucket.append(q["QPS"])
		# ucand_in_bucket.append(q["avg_cmp"])
	"""
	for q in candidate:
		print("%d %f %f"%(q["ef"],q["recall"],q["QPS"]))
		pass
	"""

print(",".join([str(q["recall"]) for q,b in q_in_bucket]))
print(",".join(["%.2f"%q["QPS"] for q,b in q_in_bucket]))
print(",".join([str(int(q["avg_cmp"])) for q,b in q_in_bucket]))

if file_out is not None:
	with open(file_out, "a", newline='') as csvfile:
		writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(["GRAPH","Parameters","Size","Build time","Avg degree","Max degree"])
		writer.writerow([
			preamble["algo"],
			"m = %d, efc = %d" % (building["m"],building["efc"]),
			preamble["scale"],
			round(building["time"], 3),
			round(building["avg_deg"], 4),
			building["max_deg"]
		]);
		writer.writerow([])
		writer.writerow([
			"Num queries","Num nonzero queries","Target recall","Actual recall","Alternative recall",
			"QPS","Average cmps","Tail cmps","Average visited","Tail visited"
		])

		for i in range(len(q_in_bucket)):
			q,b = q_in_bucket[i]
			if i+1<len(q_in_bucket) and q==q_in_bucket[i+1][0]:
				continue
			writer.writerow([
				q["cnt"],
				q["num_nz"],
				b,
				q["recall"],
				q["recall_alt"],
				round(q["QPS"], 2),
				int(q["avg_cmp"]),
				q[".99_cmp"],
				int(q["avg_visit"]),
				q[".99_visit"],
			])
		writer.writerow([])