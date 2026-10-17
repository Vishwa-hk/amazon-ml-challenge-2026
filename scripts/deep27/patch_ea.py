p="scripts/deep27/error_analysis_v6.py"
s=open(p,encoding="utf-8").read()
R=[("ROOT = Path(__file__).resolve().parents[1]","ROOT = Path(__file__).resolve().parents[2]"),
("from judge_v4 import truth_pairs, kept_entities  # noqa: E402","from judge_v4 import kept_entities  # noqa: E402\nfrom pipeline_v6 import truth as truth_pairs  # noqa: E402"),
('RUN, BL, C = ROOT / "runs" / "v4", ROOT / "runs" / "blocking_v3", ROOT / "data_cache"','RUN, BL, C = ROOT / "runs" / "v6", ROOT / "runs" / "blocking_v6", ROOT / "data_cache"'),
('"v4_error_analysis"','"v6_error_analysis"'),
('RUN / "test_like" /','RUN / "gpu_translit" /'),
('v4 stage-2 held-out','v6 stage-2 held-out'),
("sample(min(12, df.height), seed=3)","sample(min(25, df.height), seed=3)")]
for a,b in R:
    assert a in s, a
    s=s.replace(a,b)
L=s.split("\n")
i=[k for k,l in enumerate(L) if "4. EXAMPLES (train)" in l][0]
ins=['for g_, df_ in groups.items():',
     '    df_.select([c for c in df_.columns if c in ("i1","i2","p","true_owner","p_true_owner")]).write_parquet(OUT / (g_.replace(" ","_") + ".parquet"))',
     'per.write_parquet(OUT / "per_entity.parquet")']
L[i:i]=ins
open(p,"w",encoding="utf-8").write("\n".join(L))
print("patched")
