import sys
import re
import functools
import csv

p_cmd = re.compile(r"\./calc_recall -n ([0-9]*) .*?-lsh (.*?) .*?-l ([0-9]*) .*?-rot ([0-9]*)")
p_dim = re.compile(r"size: ([0-9]*)")
p_algo = re.compile(r"Start building (.*)")
p_num_hashbit = re.compile(r"num_hashbit: ([0-9]*)")
p_buildtime = re.compile(r"Build index: (.*)")
p_read_gt = re.compile(r"Read groundTruthFile")
p_qparam = re.compile(r"measure recall@([0-9]*) with num_probes=([0-9]*) on ([0-9]*) queries")
p_recall = re.compile(r"([0-9\.]*) at (.*)kqps, max_cand=([0-9]*)")
p_avg_cand = re.compile(r"average number of candidates: (.*)")
p_avg_uniq_cand = re.compile(r"average number of unique candidates: (.*)")
p_endq = re.compile(r"---")

file_in = sys.argv[1]
target_k = int(sys.argv[2]) if len(sys.argv)>2 else 0
file_out = sys.argv[3] if len(sys.argv)>3 else None

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
			building["lsh"] = res.group(2)
			building["l"] = int(res.group(3))
			building["rot"] = int(res.group(4))
			continue
		res = re.search(p_dim, l)
		if res is not None: # algorithm name
			preamble["dim"] = res.group(1)
			continue
		res = re.search(p_algo, l)
		if res is not None: # algorithm name
			preamble["algo"] = res.group(1)
			continue
		res = re.search(p_num_hashbit, l)
		if res is not None: # #hashbit
			preamble["num_hashbit"] = int(res.group(1))
			continue
		res = re.search(p_buildtime, l)
		if res is not None: # building time
			building["time"] = float(res.group(1))
			state = "BUILD"
			continue

	if state=="BUILD":
		res = re.search(p_read_gt, l)
		if res is not None: # load ground truth
			state = "QUERY"
			continue

	if state=="QUERY":
		res = re.search(p_qparam, l)
		if res is not None: # query param
			q["k"] = int(res.group(1))
			q["num_probes"] = int(res.group(2))
			q["cnt"] = int(res.group(3))
			continue
		res = re.search(p_recall, l)
		if res is not None: # recall
			q["recall"] = float(res.group(1))
			q["QPS"] = float(res.group(2))*1000
			q["max_cand"] = int(res.group(3))
			continue
		res = re.search(p_avg_cand, l)
		if res is not None:
			q["avg_cand"] = float(res.group(1))
			continue
		res = re.search(p_avg_uniq_cand, l)
		if res is not None:
			q["avg_uniq_cand"] = float(res.group(1))
			continue
		res = re.match(p_endq, l)
		if res is not None: # at the end of current query
			if target_k==0 or q["k"]==target_k:
				query.append(q)
			q = {}

f.close()

print(preamble)
print(building)
print(query[0])

query.sort(key=lambda q: q["recall"])
# print(query[0])

print("max_cand\trecall\t\tQPS\t\tucand\tnum_probes")
# bucket = [0.02,0.05,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2, 0.22,0.23,0.25,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 1]
# bucket = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 1]
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
		print("%d\t\t%f\t%f\t%d\t%d"%(q["max_cand"],q["recall"],q["QPS"],int(q["avg_uniq_cand"]),int(q["num_probes"])))
		q_in_bucket.append((q,b))
		# recall_in_bucket.append(q["recall"])
		# QPS_in_bucket.append(q["QPS"])
		# ucand_in_bucket.append(q["avg_uniq_cand"])
	"""
	for q in candidate:
		print("%d %f %f"%(q["ef"],q["recall"],q["QPS"]))
		pass
	"""

print(",".join([str(q["recall"]) for q,b in q_in_bucket]))
print(",".join(["%.2f"%q["QPS"] for q,b in q_in_bucket]))
print(",".join([str(int(q["avg_uniq_cand"])) for q,b in q_in_bucket]))

if file_out is not None:
	with open(file_out, "a", newline='') as csvfile:
		writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow(["Algorithm","Parameters","Size","Build time","Hash bits"])
		writer.writerow([
			preamble["algo"],
			"lsh = %s, l = %d, rot = %d" % (building["lsh"],building["l"],building["rot"]),
			preamble["scale"],
			round(building["time"], 3),
			preamble["num_hashbit"]
		]);
		writer.writerow([])
		writer.writerow([
			"Num queries","Target recall","Actual recall","QPS",
			"Average Cands","Average Uniq Cands","Num probes","Cand limit"
		])

		for i in range(len(q_in_bucket)):
			q,b = q_in_bucket[i]
			if i+1<len(q_in_bucket) and q==q_in_bucket[i+1][0]:
				continue
			writer.writerow([
				q["cnt"],
				b,
				q["recall"],
				round(q["QPS"], 2),
				int(q["avg_cand"]),
				int(q["avg_uniq_cand"]),
				q["num_probes"],
				q["max_cand"],
			])
		writer.writerow([])