
basePath="/iesl/canvas/luke/universal-schema-features/feature-matrix-factorization/runs/run20140507_features/"

# possible choices are "intertext" and "ngram"
#feature_types=("intertext" "ngram")
feature_types=("ngram")

# size of subset of cells of each type to use
# set to "-0" to use full dataset
subsetSize="1000"

cat "${basePath}data/test_relation.feats.cells" | awk -F$'\t' 'BEGIN{OFS="\t"}{print $1,$2,"?"}' > "${basePath}data/test_relation_marked.feats.cells"

train_feature_types=(${feature_types[@]} "relation")
test_feature_types=(${feature_types[@]} "relation_marked")

train_files=$(echo ${train_feature_types[@]} | tr ' ' '\n' | awk -v path="${basePath}data/" '{print path"train_"$1".feats.cells"}')
test_files=$(echo ${test_feature_types[@]} | tr ' ' '\n' | awk -v path="${basePath}data/" '{print path"test_"$1".feats.cells"}')

awk -v subset="$subsetSize" 'FNR<subset' $train_files > "${basePath}data/train_matrix.subset"
awk -v subset="$subsetSize" 'FNR<subset' $test_files > "${basePath}data/test_matrix.subset"