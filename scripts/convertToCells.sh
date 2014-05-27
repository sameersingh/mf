basePath="/iesl/canvas/luke/universal-schema-features/feature-matrix-factorization/runs/run20140507_features/"

getCells() {
	head -n 100 | awk -F$'\t' 'BEGIN{OFS="\t"}{split($NF,a," "); for (e in a) if (a[e]!="+1") {split(a[e],s,":"); print NR,s[1],s[2]}}';
}

for data_split in "train" "dev" "test"; do
	for feat_set in "relation" "intertext" "ngram"; do
		file_name="${data_split}_${feat_set}.feats"
		echo "Converting ${file_name}..."
		cat "${basePath}$file_name" | getCells > "${basePath}data/${file_name}.cells"
	done
done
