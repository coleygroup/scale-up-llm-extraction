import matplotlib.pyplot as plt
import pandas as pd


def plot_problemcategory_trends(
	pivot: pd.DataFrame,
	reaction_class: str,
	problem_categories: list[str],
	*,
	ax=None,
	title: str | None = None,
	marker: str = "o",
	linewidth: float = 2.0,
):
	if reaction_class not in pivot.index.get_level_values("reaction_class"):
		raise ValueError(f"reaction_class not found: {reaction_class}")

	sub = pivot.xs(reaction_class, level="reaction_class")

	years = sub.index
	try:
		years_num = years.astype(int)
		sub = sub.copy()
		sub.index = years_num
		sub = sub.sort_index()
	except Exception:
		sub = sub.sort_index()

	missing = [c for c in problem_categories if c not in sub.columns]
	cats = [c for c in problem_categories if c in sub.columns]
	if not cats:
		raise ValueError(f"None of the requested problemCategories exist for {reaction_class}. Missing: {missing}")

	if ax is None:
		fig, ax = plt.subplots(figsize=(2, 1), dpi=300)
	else:
		fig = ax.figure

	for c in cats:
		ax.plot(sub.index, sub[c].values, marker=marker, linewidth=linewidth, label=c)

	ax.set_xlabel("Year")
	ax.set_ylabel("Count")
	ax.set_title(title or f"{reaction_class}: problemCategory counts per year")
	ax.legend(frameon=False, fontsize=8)
	ax.grid(True, alpha=0.25)

	if missing:
		print(f"Note: missing categories (not present in columns): {missing}")

	return fig, ax


def find_problem_split_year(
	pivot: pd.DataFrame,
	reaction_class: str,
	problem_category: str,
	*,
	score_metric: str = "mean_gap",
	exclude_last_year: bool = True,
	min_years_before: int = 2,
	min_years_after: int = 2,
):
	if reaction_class not in pivot.index.get_level_values("reaction_class"):
		raise ValueError(f"reaction_class not found: {reaction_class}")

	sub = pivot.xs(reaction_class, level="reaction_class")

	years = sub.index
	try:
		years_num = years.astype(int)
		sub = sub.copy()
		sub.index = years_num
		sub = sub.sort_index()
	except Exception:
		sub = sub.sort_index()

	if problem_category not in sub.columns:
		raise ValueError(f"problem_category not found for {reaction_class}: {problem_category}")

	counts = sub[problem_category].fillna(0).astype(float)
	total = counts.sum()
	if total == 0:
		raise ValueError(f"No occurrences found for {problem_category} in {reaction_class}")

	before = counts.cumsum()
	after = total - before
	n_years = len(counts)
	n_before = pd.Series(range(1, n_years + 1), index=counts.index, dtype=float)
	n_after = float(n_years) - n_before

	mean_before = before / n_before
	mean_after = after / n_after.where(n_after > 0)
	difference = before - after
	mean_gap = mean_before - mean_after

	split_table = pd.DataFrame(
		{
			"count": counts,
			"before_inclusive": before,
			"after_exclusive": after,
			"n_years_before": n_before,
			"n_years_after": n_after,
			"mean_before": mean_before,
			"mean_after": mean_after,
			"difference": difference,
			"mean_gap": mean_gap,
		}
	)

	valid = (split_table["n_years_before"] >= float(min_years_before)) & (
		split_table["n_years_after"] >= float(min_years_after)
	)
	if exclude_last_year:
		valid = valid & (split_table["n_years_after"] >= 1)

	candidate_table = split_table[valid].copy()
	if candidate_table.empty:
		raise ValueError(
			"No valid split years after applying min_years_before/min_years_after constraints"
		)

	if score_metric not in candidate_table.columns:
		raise ValueError(f"Unknown score_metric: {score_metric}")

	best_year = candidate_table[score_metric].idxmax()
	best_score = float(candidate_table.loc[best_year, score_metric])

	return best_year, best_score, split_table


def _split_summary_columns() -> list[str]:
	return [
		"reaction_class",
		"problem_category",
		"best_year",
		"split_score",
		"score_metric",
		"total_count",
		"before_inclusive_at_best",
		"after_exclusive_at_best",
		"mean_before_at_best",
		"mean_after_at_best",
		"n_years_before_at_best",
		"n_years_after_at_best",
	]


def summarize_problem_split_years_for_reaction_class(
	pivot: pd.DataFrame,
	reaction_class: str,
	*,
	problem_categories: list[str] | None = None,
	score_metric: str = "mean_gap",
	exclude_last_year: bool = True,
	min_years_before: int = 2,
	min_years_after: int = 2,
	min_total_occurrences: float = 1.0,
) -> pd.DataFrame:
	if reaction_class not in pivot.index.get_level_values("reaction_class"):
		raise ValueError(f"reaction_class not found: {reaction_class}")

	sub = pivot.xs(reaction_class, level="reaction_class")

	years = sub.index
	try:
		years_num = years.astype(int)
		sub = sub.copy()
		sub.index = years_num
		sub = sub.sort_index()
	except Exception:
		sub = sub.sort_index()

	min_required_years = int(min_years_before) + int(min_years_after)
	if len(sub.index) < min_required_years:
		return pd.DataFrame(columns=_split_summary_columns())

	if problem_categories is None:
		cats = list(sub.columns)
	else:
		cats = [c for c in problem_categories if c in sub.columns]

	rows = []
	for problem_category in cats:
		total = float(sub[problem_category].fillna(0).sum())
		if total < float(min_total_occurrences):
			continue

		try:
			best_year, best_score, split_table = find_problem_split_year(
				pivot,
				reaction_class=reaction_class,
				problem_category=problem_category,
				score_metric=score_metric,
				exclude_last_year=exclude_last_year,
				min_years_before=min_years_before,
				min_years_after=min_years_after,
			)
		except ValueError:
			continue

		rows.append(
			{
				"reaction_class": reaction_class,
				"problem_category": problem_category,
				"best_year": best_year,
				"split_score": best_score,
				"score_metric": score_metric,
				"total_count": total,
				"before_inclusive_at_best": float(split_table.loc[best_year, "before_inclusive"]),
				"after_exclusive_at_best": float(split_table.loc[best_year, "after_exclusive"]),
				"mean_before_at_best": float(split_table.loc[best_year, "mean_before"]),
				"mean_after_at_best": float(split_table.loc[best_year, "mean_after"]),
				"n_years_before_at_best": int(split_table.loc[best_year, "n_years_before"]),
				"n_years_after_at_best": int(split_table.loc[best_year, "n_years_after"]),
			}
		)

	if not rows:
		return pd.DataFrame(columns=_split_summary_columns())

	out = pd.DataFrame(rows).sort_values(
		["split_score", "total_count"],
		ascending=[False, False],
	)
	return out.reset_index(drop=True)


def summarize_problem_split_years_all_reaction_classes(
	pivot: pd.DataFrame,
	*,
	reaction_classes: list[str] | None = None,
	problem_categories: list[str] | None = None,
	score_metric: str = "mean_gap",
	exclude_last_year: bool = True,
	min_years_before: int = 2,
	min_years_after: int = 2,
	min_total_occurrences: float = 1.0,
) -> pd.DataFrame:
	if reaction_classes is None:
		reaction_classes = list(pd.Index(pivot.index.get_level_values("reaction_class")).unique())

	all_rows = []
	for reaction_class in reaction_classes:
		rc_table = summarize_problem_split_years_for_reaction_class(
			pivot,
			reaction_class=reaction_class,
			problem_categories=problem_categories,
			score_metric=score_metric,
			exclude_last_year=exclude_last_year,
			min_years_before=min_years_before,
			min_years_after=min_years_after,
			min_total_occurrences=min_total_occurrences,
		)
		if len(rc_table) > 0:
			all_rows.append(rc_table)

	if not all_rows:
		return pd.DataFrame(columns=_split_summary_columns())

	out = pd.concat(all_rows, ignore_index=True)
	out = out.sort_values(["split_score", "total_count"], ascending=[False, False])
	return out.reset_index(drop=True)


def plot_split_entry_distribution(
	pivot: pd.DataFrame,
	split_row: pd.Series | dict,
	*,
	ax=None,
	marker: str = "o",
	linewidth: float = 2.0,
	bar_alpha: float = 0.35,
):
	row = split_row if isinstance(split_row, pd.Series) else pd.Series(split_row)

	reaction_class = str(row["reaction_class"])
	problem_category = row["problem_category"]

	if reaction_class not in pivot.index.get_level_values("reaction_class"):
		raise ValueError(f"reaction_class not found: {reaction_class}")

	sub = pivot.xs(reaction_class, level="reaction_class")

	years = sub.index
	try:
		years_num = years.astype(int)
		sub = sub.copy()
		sub.index = years_num
		sub = sub.sort_index()
	except Exception:
		sub = sub.sort_index()

	if problem_category not in sub.columns:
		raise ValueError(f"problem_category not found for {reaction_class}: {problem_category}")

	counts = sub[problem_category].fillna(0).astype(float)

	if ax is None:
		fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
	else:
		fig = ax.figure

	ax.bar(counts.index, counts.values, alpha=bar_alpha, label="count")
	ax.plot(counts.index, counts.values, marker=marker, linewidth=linewidth, color="C0")

	if "best_year" in row and pd.notna(row["best_year"]):
		split_year = int(row["best_year"])
		ax.axvline(split_year, color="C3", linestyle="--", linewidth=1.5, label=f"best_year={split_year}")

	score_metric = row.get("score_metric", "score")
	score_value = row.get("split_score", None)
	title = f"{reaction_class} | {problem_category}"
	if pd.notna(score_value):
		title += f" | {score_metric}={float(score_value):.3f}"

	ax.set_title(title)
	ax.set_xlabel("Year")
	ax.set_ylabel("Count")
	ax.grid(True, alpha=0.25)
	ax.legend(frameon=False, fontsize=8)

	year_table = pd.DataFrame({"count": counts})
	return fig, ax, year_table


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
	for col in candidates:
		if col in df.columns:
			return col
	return None


def find_shift_exemplar_reaction(
	df_long: pd.DataFrame,
	split_row: pd.Series | dict,
	*,
	reaction_col_candidates: list[str] | None = None,
	paper_col_candidates: list[str] | None = None,
	year_col: str = "pub_year",
) -> dict:
	row = split_row if isinstance(split_row, pd.Series) else pd.Series(split_row)

	reaction_col_candidates = reaction_col_candidates or [
		"reaction_smiles", "rxn_smiles", "reaction", "smiles", "reaction_id", "rxn_id"
	]
	paper_col_candidates = paper_col_candidates or [
		"source_folder", "paper_id", "paper", "source", "doi", "id"
	]

	required = ["reaction_class", "problem_category", "best_year"]
	missing = [k for k in required if k not in row.index]
	if missing:
		raise ValueError(f"split_row missing required fields: {missing}")

	reaction_class = str(row["reaction_class"])
	problem_category = row["problem_category"]
	split_year = int(row["best_year"] if pd.notna(row["best_year"]) else 0)
	split_score = float(row.get("split_score", 0.0))

	if "reaction_class" not in df_long.columns or "problemCategory" not in df_long.columns:
		raise ValueError("df_long must contain reaction_class and problemCategory columns")
	if year_col not in df_long.columns:
		raise ValueError(f"df_long missing year column: {year_col}")

	reaction_col = _pick_existing_column(df_long, reaction_col_candidates)
	paper_col = _pick_existing_column(df_long, paper_col_candidates)

	sub = df_long[
		(df_long["reaction_class"] == reaction_class)
		& (df_long["problemCategory"] == problem_category)
	].copy()

	if len(sub) == 0:
		return {
			"reaction_class": reaction_class,
			"problem_category": problem_category,
			"best_year": split_year,
			"shift_side": None,
			"paper_col": paper_col,
			"paper_value": None,
			"reaction_col": reaction_col,
			"reaction_value": None,
			"year": None,
			"n_candidates": 0,
		}

	sub["_year_num"] = pd.to_numeric(sub[year_col], errors="coerce")
	sub = sub[sub["_year_num"].notna()].copy()
	sub["_year_num"] = sub["_year_num"].astype(int)

	side = "before" if split_score >= 0 else "after"
	if side == "before":
		cands = sub[sub["_year_num"] <= split_year].copy()
	else:
		cands = sub[sub["_year_num"] > split_year].copy()

	if len(cands) == 0:
		cands = sub.copy()
		side = "all"

	if paper_col is not None:
		paper_counts = cands.groupby(paper_col).size().sort_values(ascending=False)
		top_paper = paper_counts.index[0]
		cands = cands[cands[paper_col] == top_paper].copy()

	cands["_dist_to_split"] = (cands["_year_num"] - split_year).abs()

	if reaction_col is not None:
		rxn_text = cands[reaction_col].astype(str).str.strip()
		cands["_has_rxn"] = (rxn_text != "") & (rxn_text.str.lower() != "nan")
	else:
		cands["_has_rxn"] = False

	if side == "before":
		cands = cands.sort_values(["_has_rxn", "_dist_to_split", "_year_num"], ascending=[False, True, False])
	else:
		cands = cands.sort_values(["_has_rxn", "_dist_to_split", "_year_num"], ascending=[False, True, True])

	best = cands.iloc[0]

	return {
		"reaction_class": reaction_class,
		"problem_category": problem_category,
		"best_year": split_year,
		"shift_side": side,
		"split_score": split_score,
		"paper_col": paper_col,
		"paper_value": (None if paper_col is None else best[paper_col]),
		"reaction_col": reaction_col,
		"reaction_value": (None if reaction_col is None else best[reaction_col]),
		"year": int(best["_year_num"]),
		"n_candidates": int(len(cands)),
	}


def add_shift_exemplar_reactions(
	df_long: pd.DataFrame,
	split_table: pd.DataFrame,
) -> pd.DataFrame:
	if len(split_table) == 0:
		return split_table.copy()

	exemplar_rows = []
	for _, r in split_table.iterrows():
		exemplar_rows.append(find_shift_exemplar_reaction(df_long, r))

	exemplar_df = pd.DataFrame(exemplar_rows)
	out = split_table.reset_index(drop=True).join(
		exemplar_df[["shift_side", "paper_col", "paper_value", "reaction_col", "reaction_value", "year"]]
	)
	return out


def analyze_solution_distribution_at_pivot(
	df_long: pd.DataFrame,
	split_row: pd.Series | dict,
	*,
	solution_col: str = "solutionCategory",
	year_col: str = "pub_year",
	reaction_class_col: str = "reaction_class",
	problem_col: str = "problemCategory",
	min_total_mentions: int = 1,
) -> tuple[pd.DataFrame, dict]:
	row = split_row if isinstance(split_row, pd.Series) else pd.Series(split_row)

	required = ["reaction_class", "problem_category", "best_year"]
	missing = [k for k in required if k not in row.index]
	if missing:
		raise ValueError(f"split_row missing required fields: {missing}")

	reaction_class = str(row["reaction_class"])
	problem_category = row["problem_category"]
	pivot_year = int(row["best_year"] if pd.notna(row["best_year"]) else 0)

	needed_cols = [reaction_class_col, problem_col, solution_col, year_col]
	missing_cols = [c for c in needed_cols if c not in df_long.columns]
	if missing_cols:
		raise ValueError(f"df_long missing required columns: {missing_cols}")

	sub = df_long[
		(df_long[reaction_class_col] == reaction_class)
		& (df_long[problem_col] == problem_category)
	][needed_cols].copy()

	sub["_year_num"] = pd.to_numeric(sub[year_col], errors="coerce")
	sub = sub[sub["_year_num"].notna()].copy()
	sub["_year_num"] = sub["_year_num"].astype(int)

	before = sub[sub["_year_num"] <= pivot_year].copy()
	after = sub[sub["_year_num"] > pivot_year].copy()

	before_counts = before[solution_col].value_counts(dropna=False)
	after_counts = after[solution_col].value_counts(dropna=False)

	all_solutions = sorted(set(before_counts.index).union(set(after_counts.index)), key=lambda x: str(x))
	dist = pd.DataFrame(index=all_solutions)
	dist["count_before"] = before_counts.reindex(all_solutions, fill_value=0).astype(int)
	dist["count_after"] = after_counts.reindex(all_solutions, fill_value=0).astype(int)
	dist["count_total"] = dist["count_before"] + dist["count_after"]

	dist = dist[dist["count_total"] >= int(min_total_mentions)].copy()

	total_before = int(dist["count_before"].sum())
	total_after = int(dist["count_after"].sum())

	dist["share_before"] = (dist["count_before"] / total_before) if total_before > 0 else 0.0
	dist["share_after"] = (dist["count_after"] / total_after) if total_after > 0 else 0.0
	dist["delta_share"] = dist["share_after"] - dist["share_before"]
	dist["fold_change_count"] = (dist["count_after"] + 1.0) / (dist["count_before"] + 1.0)

	dist = dist.sort_values(["delta_share", "count_after"], ascending=[False, False]).reset_index()
	dist = dist.rename(columns={"index": solution_col})

	n_increased = int((dist["delta_share"] > 0).sum())
	top_increase = None
	if len(dist) > 0:
		top_row = dist.iloc[0]
		top_increase = {
			"solution": top_row[solution_col],
			"delta_share": float(top_row["delta_share"]),
			"share_before": float(top_row["share_before"]),
			"share_after": float(top_row["share_after"]),
			"count_before": int(top_row["count_before"]),
			"count_after": int(top_row["count_after"]),
		}

	summary = {
		"reaction_class": reaction_class,
		"problem_category": problem_category,
		"pivot_year": pivot_year,
		"n_rows_matched": int(len(sub)),
		"n_before_rows": int(len(before)),
		"n_after_rows": int(len(after)),
		"n_solution_classes": int(len(dist)),
		"n_solution_classes_increased": n_increased,
		"top_increase": top_increase,
	}

	return dist, summary


def assess_solution_increase_at_pivot(
	dist: pd.DataFrame,
	*,
	solution_col: str = "solutionCategory",
	min_delta_share: float = 0.03,
	min_after_count: int = 3,
) -> pd.DataFrame:
	if len(dist) == 0:
		return dist.copy()

	keep = dist[
		(dist["delta_share"] >= float(min_delta_share))
		& (dist["count_after"] >= int(min_after_count))
	].copy()

	return keep.sort_values(["delta_share", "count_after"], ascending=[False, False]).reset_index(drop=True)


def find_solution_increase_exemplar_reaction(
	df_long: pd.DataFrame,
	split_row: pd.Series | dict,
	solution_value,
	*,
	solution_col: str = "solutionCategory",
	year_col: str = "pub_year",
	reaction_class_col: str = "reaction_class",
	problem_col: str = "problemCategory",
	reaction_col_candidates: list[str] | None = None,
	paper_col_candidates: list[str] | None = None,
) -> dict:
	row = split_row if isinstance(split_row, pd.Series) else pd.Series(split_row)

	reaction_col_candidates = reaction_col_candidates or [
		"reaction_smiles", "rxn_smiles", "reaction", "smiles", "reaction_id", "rxn_id"
	]
	paper_col_candidates = paper_col_candidates or [
		"source_folder", "paper_id", "paper", "source", "doi", "id"
	]

	required = ["reaction_class", "problem_category", "best_year"]
	missing = [k for k in required if k not in row.index]
	if missing:
		raise ValueError(f"split_row missing required fields: {missing}")

	needed_cols = [reaction_class_col, problem_col, solution_col, year_col]
	missing_cols = [c for c in needed_cols if c not in df_long.columns]
	if missing_cols:
		raise ValueError(f"df_long missing required columns: {missing_cols}")

	reaction_class = str(row["reaction_class"])
	problem_category = row["problem_category"]
	split_year = int(row["best_year"] if pd.notna(row["best_year"]) else 0)

	reaction_col = _pick_existing_column(df_long, reaction_col_candidates)
	paper_col = _pick_existing_column(df_long, paper_col_candidates)

	sub = df_long[
		(df_long[reaction_class_col] == reaction_class)
		& (df_long[problem_col] == problem_category)
		& (df_long[solution_col] == solution_value)
	].copy()

	if len(sub) == 0:
		return {
			"solution_value": solution_value,
			"paper_col": paper_col,
			"paper_value": None,
			"reaction_col": reaction_col,
			"reaction_value": None,
			"exemplar_year": None,
			"n_candidates": 0,
		}

	sub["_year_num"] = pd.to_numeric(sub[year_col], errors="coerce")
	sub = sub[sub["_year_num"].notna()].copy()
	sub["_year_num"] = sub["_year_num"].astype(int)

	after = sub[sub["_year_num"] > split_year].copy()
	cands = after if len(after) > 0 else sub.copy()

	if paper_col is not None and len(cands) > 0:
		paper_counts = cands.groupby(paper_col).size().sort_values(ascending=False)
		top_paper = paper_counts.index[0]
		cands = cands[cands[paper_col] == top_paper].copy()

	cands["_dist_to_split"] = (cands["_year_num"] - split_year).abs()

	if reaction_col is not None:
		rxn_text = cands[reaction_col].astype(str).str.strip()
		cands["_has_rxn"] = (rxn_text != "") & (rxn_text.str.lower() != "nan")
	else:
		cands["_has_rxn"] = False

	cands = cands.sort_values(["_has_rxn", "_dist_to_split", "_year_num"], ascending=[False, True, True])
	best = cands.iloc[0]

	return {
		"solution_value": solution_value,
		"paper_col": paper_col,
		"paper_value": (None if paper_col is None else best[paper_col]),
		"reaction_col": reaction_col,
		"reaction_value": (None if reaction_col is None else best[reaction_col]),
		"exemplar_year": int(best["_year_num"]),
		"n_candidates": int(len(cands)),
	}


def link_solution_increases_to_exemplar_reactions(
	df_long: pd.DataFrame,
	split_row: pd.Series | dict,
	increases_df: pd.DataFrame,
	*,
	solution_col: str = "solutionCategory",
) -> pd.DataFrame:
	if solution_col not in increases_df.columns:
		raise ValueError(f"increases_df missing solution column: {solution_col}")

	rows = []
	for _, inc in increases_df.iterrows():
		solution_value = inc[solution_col]
		exemplar = find_solution_increase_exemplar_reaction(
			df_long,
			split_row,
			solution_value,
			solution_col=solution_col,
		)

		merged = inc.to_dict()
		merged.update(exemplar)
		rows.append(merged)

	if not rows:
		return pd.DataFrame(columns=list(increases_df.columns) + [
			"paper_col", "paper_value", "reaction_col", "reaction_value", "exemplar_year", "n_candidates"
		])

	out = pd.DataFrame(rows)
	sort_cols = [c for c in ["delta_share", "count_after"] if c in out.columns]
	if sort_cols:
		ascending = [False] * len(sort_cols)
		out = out.sort_values(sort_cols, ascending=ascending)
	return out.reset_index(drop=True)


def plot_linked_solution_increases(
	linked_df: pd.DataFrame,
	*,
	solution_col: str = "solutionCategory",
	delta_col: str = "delta_share",
	paper_col: str = "paper_value",
	top_n: int = 10,
	figsize: tuple[float, float] = (10, 6),
) -> tuple[object, object, pd.DataFrame]:
	required = [solution_col, delta_col]
	missing = [c for c in required if c not in linked_df.columns]
	if missing:
		raise ValueError(f"linked_df missing required columns: {missing}")

	if len(linked_df) == 0:
		raise ValueError("linked_df is empty")

	plot_df = linked_df.sort_values(delta_col, ascending=False).head(int(top_n)).copy()
	plot_df = plot_df.iloc[::-1].copy()

	if paper_col in plot_df.columns:
		plot_df["_label"] = plot_df[solution_col].astype(str) + " | " + plot_df[paper_col].astype(str)
	else:
		plot_df["_label"] = plot_df[solution_col].astype(str)

	fig, ax = plt.subplots(figsize=figsize, dpi=150)
	ax.barh(plot_df["_label"], plot_df[delta_col], alpha=0.8, color="C0")
	ax.axvline(0, color="black", linewidth=1)
	ax.set_xlabel("Increase in share after pivot (delta_share)")
	ax.set_ylabel("Solution class | exemplar source")
	ax.set_title(f"Top {min(int(top_n), len(linked_df))} linked post-pivot solution increases")
	ax.grid(True, axis="x", alpha=0.25)

	return fig, ax, plot_df
