"""Benchmark GLIPH control tokenization cost and rare-token coverage.

Runs only when ``RUN_BENCHMARK=1``.

        # --- batch all families, with mappings (study mode, for comparison) ---
        _study_art, study_stats = _measure_extraction(rep, list(DEFAULT_FAMILIES), count_mode="clonotype", build_mappings=True)
        rows.append(
            {
                "size": n,
                "mode": "batch_study",
                "family": "all",
                "elapsed_s": float(study_stats["elapsed_s"]),
                "peak_mb": float(study_stats["peak_mb"]),
                "tokens_total": int(study_stats["tokens_total"]),
                "single_family_total_s": np.nan,
                "single_family_peak_mb": np.nan,
                "speedup_vs_single_total": np.nan,
            }
        )

        benchmark_log_line(
            "GLIPH tokenization scale "
            f"n={n}: ctrl_batch_s={batch_stats['elapsed_s']:.3f}, "
            f"study_batch_s={study_stats['elapsed_s']:.3f}, "
            f"single_total_s={per_family_elapsed:.3f}, "
            f"ctrl_peak_mb={batch_stats['peak_mb']:.1f}, study_peak_mb={study_stats['peak_mb']:.1f}"
        )

    result_df = pd.DataFrame(rows)
    ctrl_df = result_df[result_df["mode"] == "batch_ctrl"].copy()
    study_df_bm = result_df[result_df["mode"] == "batch_study"].copy()

    print("\nGLIPH control tokenization benchmark (per family + batch):")
    print(result_df.to_string(index=False))
    print("\nControl-mode (counts-only) batch summary by control size:")
    print(ctrl_df[["size", "elapsed_s", "single_family_total_s", "speedup_vs_single_total", "peak_mb", "tokens_total"]].to_string(index=False))
    print("\nStudy-mode (with mappings) batch summary by control size:")
    print(study_df_bm[["size", "elapsed_s", "peak_mb", "tokens_total"]].to_string(index=False))
    if not ctrl_df.empty and not study_df_bm.empty:
        merged = ctrl_df[["size", "elapsed_s", "peak_mb"]].merge(
            study_df_bm[["size", "elapsed_s", "peak_mb"]],
            on="size", suffixes=("_ctrl", "_study"),
        )
        merged["speedup_ctrl_vs_study"] = merged["elapsed_s_study"] / merged["elapsed_s_ctrl"]
        merged["mem_reduction"] = merged["peak_mb_study"] / merged["peak_mb_ctrl"]
        print("\nControl vs study mode comparison (speedup = study_time / ctrl_time):")
        print(merged.to_string(index=False))
GLIPH_PATH = Path(__file__).resolve().parents[1] / "airr_benchmark" / "gliph" / "gliph_trb.tsv.gz"
    assert not ctrl_df.empty
    assert set(ctrl_df["size"]) == set(sizes)
    assert ctrl_df["tokens_total"].gt(0).all()
DEFAULT_FAMILIES = ("v3", "pos3", "u3", "u4", "g4", "g5")
AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
SEED = 42


def _control_sizes() -> list[int]:
    raw = os.getenv("MIRPY_GLPH_CONTROL_SIZES", "100000,1000000,10000000")
    values: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            value = int(tok)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    if not values:
        values = [100_000, 1_000_000, 10_000_000]
    return sorted(set(values))


def _token_threads() -> int:
    raw = os.getenv("MIRPY_GLPH_TOKEN_THREADS")
    if raw is None:
        return 8
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(1, value)


def _canonical_control_df() -> pd.DataFrame:
    ctrl_raw = ControlManager().ensure_and_load_control_df("real", "human", "TRB")
    df = pd.DataFrame(
        {
            "junction_aa": ctrl_raw["junction_aa"].astype(str).str.strip(),
            "v_gene": ctrl_raw["v_gene"].astype(str).str.strip(),
            "j_gene": ctrl_raw["j_gene"].astype(str).str.strip(),
            "duplicate_count": pd.to_numeric(ctrl_raw.get("duplicate_count", 1), errors="coerce").fillna(1).astype(int),
        }
    )
    mask = df["junction_aa"].str.len().ge(5) & df["junction_aa"].str.match(AA_RE)
    df = df.loc[mask].reset_index(drop=True)
    return df


def _sample_to_repertoire(control_df: pd.DataFrame, idx: np.ndarray) -> LocusRepertoire:
    sampled = control_df.iloc[idx].copy()
    sampled.insert(0, "sequence_id", np.arange(len(sampled), dtype=np.int64).astype(str))
    pl_df = pl.from_pandas(sampled, include_index=False)
    return LocusRepertoire.from_polars(pl_df, locus="TRB")


def _measure_extraction(
    repertoire: LocusRepertoire,
    families: list[str],
    *,
    count_mode: str,
    build_mappings: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    artifacts = extract_gliph_artifacts_batch_from_repertoire(
        repertoire,
        families,
        count_mode=count_mode,
        build_mappings=build_mappings,
    )
    elapsed_s = time.perf_counter() - t0
    _curr, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary = {
        "families": list(families),
        "elapsed_s": elapsed_s,
        "peak_mb": peak_bytes / (1024.0 * 1024.0),
        "tokens_total": int(sum(len(art.counts) for art in artifacts.values())),
    }
    return artifacts, summary


def _normalize_gliph_df(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "junction_aa": raw["junction_aa"].astype(str).str.strip(),
            "v_gene": raw["v_gene"].astype(str).str.strip(),
            "j_gene": raw["j_gene"].astype(str).str.strip(),
            "duplicate_count": pd.to_numeric(raw["duplicate_count"], errors="coerce").fillna(1).astype(int),
            "reference_id": raw["reference_id"].astype(str).str.strip(),
            "stimulus": raw["stimulus"].astype(str).str.strip(),
            "epitope": raw["epitope"].astype(str).str.strip(),
            "gliph_cluster_id": raw["gliph_cluster_id"].astype(str).str.strip(),
        }
    )
    out = out[out["junction_aa"].str.len() >= 5].copy()
    out = out[out["junction_aa"].str.match(AA_RE)].copy()
    return deduplicate_clonotype_rows(out, subset=("reference_id", "v_gene", "junction_aa"))


def _zipf_fit(counts: dict[str, int]) -> dict[str, float]:
    freqs = np.array(sorted((int(v) for v in counts.values() if int(v) > 0), reverse=True), dtype=float)
    if len(freqs) < 5:
        return {"zipf_slope": np.nan, "zipf_r2": np.nan}

    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    x = np.log(ranks)
    y = np.log(freqs)
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"zipf_slope": float(slope), "zipf_r2": float(r2)}


@skip_benchmarks
@pytest.mark.very_slow_benchmark
def test_gliph_control_tokenization_scale_benchmark() -> None:
    sizes = _control_sizes()
    threads = _token_threads()

    control_df = _canonical_control_df()
    max_n = max(sizes)
    if max_n > len(control_df):
        pytest.skip(f"Requested max size {max_n} exceeds available control clonotypes {len(control_df)}")

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(control_df), size=max_n, replace=False)

    rows: list[dict[str, object]] = []
    benchmark_log_line(
        f"GLIPH control tokenization benchmark start: sizes={sizes}, families={list(DEFAULT_FAMILIES)}, threads={threads}"
    )

    for n in sizes:
        idx = np.sort(chosen[:n])
        rep = _sample_to_repertoire(control_df, idx)

        # --- single family, counts-only (control mode) ---
        per_family_elapsed = 0.0
        per_family_peak = 0.0
        for family in DEFAULT_FAMILIES:
            _art, stats = _measure_extraction(rep, [family], count_mode="clonotype", build_mappings=False)
            per_family_elapsed += float(stats["elapsed_s"])
            per_family_peak = max(per_family_peak, float(stats["peak_mb"]))
            rows.append(
                {
                    "size": n,
                    "mode": "single_family_ctrl",
                    "family": family,
                    "elapsed_s": float(stats["elapsed_s"]),
                    "peak_mb": float(stats["peak_mb"]),
                    "tokens_total": int(stats["tokens_total"]),
                }
            )

        # --- batch all families, counts-only (control mode) ---
        _batch_art, batch_stats = _measure_extraction(rep, list(DEFAULT_FAMILIES), count_mode="clonotype", build_mappings=False)
        rows.append(
            {
                "size": n,
                "mode": "batch_ctrl",
                "family": "all",
                "elapsed_s": float(batch_stats["elapsed_s"]),
                "peak_mb": float(batch_stats["peak_mb"]),
                "tokens_total": int(batch_stats["tokens_total"]),
                "single_family_total_s": per_family_elapsed,
                "single_family_peak_mb": per_family_peak,
                "speedup_vs_single_total": (
                    per_family_elapsed / float(batch_stats["elapsed_s"])
                    if float(batch_stats["elapsed_s"]) > 0
                    else np.nan
                ),
            }
        )



@skip_benchmarks
@pytest.mark.very_slow_benchmark
def test_gliph_rare_token_coverage_vs_control_size_benchmark() -> None:
    if not GLIPH_PATH.exists():
        pytest.skip(f"Missing GLIPH dataset: {GLIPH_PATH}")

    sizes = _control_sizes()
    control_df = _canonical_control_df()
    max_n = max(sizes)
    if max_n > len(control_df):
        pytest.skip(f"Requested max size {max_n} exceeds available control clonotypes {len(control_df)}")

    raw = pd.read_csv(GLIPH_PATH, sep="\t")
    study_df = _normalize_gliph_df(raw)

    study_pl = pl.from_pandas(
        study_df[["row_id", "junction_aa", "v_gene", "j_gene", "duplicate_count"]]
        .rename(columns={"row_id": "sequence_id"}),
        include_index=False,
    )
    study_rep = LocusRepertoire.from_polars(study_pl, locus="TRB")
    study_artifacts = extract_gliph_artifacts_batch_from_repertoire(
        study_rep,
        list(DEFAULT_FAMILIES),
        count_mode="clonotype",
    )

    rare_sets: dict[str, dict[str, set[str]]] = {}
    zipf_rows: list[dict[str, object]] = []
    for family, art in study_artifacts.items():
        counts = {k: int(v) for k, v in art.clonotype_counts.items()}
        rare_sets[family] = {
            "n1": {tok for tok, c in counts.items() if c == 1},
            "n2": {tok for tok, c in counts.items() if c == 2},
            "n3p": {tok for tok, c in counts.items() if c >= 3},
        }
        fit = _zipf_fit(counts)
        zipf_rows.append(
            {
                "family": family,
                "n_tokens": len(counts),
                "zipf_slope": fit["zipf_slope"],
                "zipf_r2": fit["zipf_r2"],
            }
        )

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(control_df), size=max_n, replace=False)

    rows: list[dict[str, object]] = []
    for n in sizes:
        idx = np.sort(chosen[:n])
        rep = _sample_to_repertoire(control_df, idx)
        ctrl_artifacts = extract_gliph_artifacts_batch_from_repertoire(
            rep,
            list(DEFAULT_FAMILIES),
            count_mode="clonotype",
        )

        for family in DEFAULT_FAMILIES:
            ctrl_tokens = set(ctrl_artifacts[family].counts)
            for bucket, target_tokens in rare_sets[family].items():
                if not target_tokens:
                    missing = 0
                    missing_frac = np.nan
                    coverage = np.nan
                else:
                    missing = len(target_tokens - ctrl_tokens)
                    missing_frac = missing / len(target_tokens)
                    coverage = 1.0 - missing_frac

                rows.append(
                    {
                        "size": n,
                        "family": family,
                        "bucket": bucket,
                        "target_tokens": len(target_tokens),
                        "missing_tokens": missing,
                        "missing_fraction": missing_frac,
                        "coverage": coverage,
                    }
                )

    coverage_df = pd.DataFrame(rows)
    zipf_df = pd.DataFrame(zipf_rows).sort_values("family")

    print("\nGLIPH token-frequency Zipf fit by family:")
    print(zipf_df.to_string(index=False))
    print("\nRare-token coverage vs control size:")
    print(coverage_df.sort_values(["size", "family", "bucket"]).to_string(index=False))

    aggregate = (
        coverage_df.groupby(["size", "bucket"], as_index=False)
        .agg(target_tokens=("target_tokens", "sum"), missing_tokens=("missing_tokens", "sum"))
        .assign(
            missing_fraction=lambda df_: np.where(
                df_["target_tokens"] > 0,
                df_["missing_tokens"] / df_["target_tokens"],
                np.nan,
            )
        )
    )
    print("\nAggregate missing fraction across families:")
    print(aggregate.to_string(index=False))

    benchmark_log_line("GLIPH rare token coverage benchmark complete")

    assert not coverage_df.empty
    assert set(coverage_df["size"]) == set(sizes)
    for family in DEFAULT_FAMILIES:
        fam = coverage_df[coverage_df["family"] == family].copy()
        for bucket in ("n1", "n2", "n3p"):
            sub = fam[fam["bucket"] == bucket].sort_values("size")
            vals = [v for v in sub["missing_tokens"].tolist() if pd.notna(v)]
            if len(vals) >= 2:
                assert all(later <= earlier for earlier, later in zip(vals[:-1], vals[1:]))
