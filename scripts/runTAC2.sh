#basePath="/iesl/canvas/luke/universal-schema-features/feature-matrix-factorization/runs/run20140507_features/"
basePath="/home/beroth/canvas/workspace/feature-matrix-factorization/runs/run20140507_features/"

# always use relation feats at train and not at test
use_intertext=true
use_ngrams=false

getCells() {
	#awk -F$'\t' 'BEGIN{OFS="\t"}{split($NF,a," "); for (e in a) if (a[e]!="+1") {split(a[e],s,":"); print NR,s[1],s[2]}}';
	awk -F$'\t' 'BEGIN{OFS="\t"}{split($NF,a," "); for (e in a) if (a[e]!="+1") {split(a[e],s,":"); print NR,s[1],"1"}}';
}
export -f getCells

echo "relation cells ..."
cat "${basePath}train_relation.feats" \
| getCells \
> "${basePath}train.mtx"

head -1 "${basePath}train.mtx"

cat "${basePath}dev_relation.feats" \
| getCells \
| awk -F$'\t' 'BEGIN{OFS="\t"}{print $1,$2,"?"}' \
> "${basePath}dev.mtx"

head -1 "${basePath}dev.mtx"

cat "${basePath}test_relation.feats" \
| getCells \
| awk -F$'\t' 'BEGIN{OFS="\t"}{print $1,$2,"?"}' \
> "${basePath}test.mtx"

head -1 "${basePath}test.mtx"

echo "... done."

if [ $use_intertext = true ]; then
	echo "intertext cells ..."
	cat "${basePath}train_intertext.feats" \
	| getCells \
	>> "${basePath}train.mtx"
        cat "${basePath}dev_intertext.feats" \
        | getCells \
        >> "${basePath}dev.mtx"
	cat "${basePath}test_intertext.feats" \
	| getCells \
	>> "${basePath}test.mtx"
	echo "... done."
fi

if [ $use_ngrams = true ]; then
	echo "ngram cells ..."
        cat "${basePath}train_ngram.feats" \
        | getCells \
        >> "${basePath}train.mtx"
        cat "${basePath}dev_ngram.feats" \
        | getCells \
        >> "${basePath}dev.mtx"
        cat "${basePath}test_ngram.feats" \
        | getCells \
        >> "${basePath}test.mtx"
	echo "... done."
fi

# run embedding

dimensionSize="50"
lambda="0.1"
iterations="10"
baseRate="0.01"
logEvery="1000000"

sharedOptionsString="--embedding-size=$dimensionSize --lambda=$lambda --iterations=$iterations --base-rate=$baseRate --log-every=$logEvery"

trainMatrix="${basePath}train.mtx"
colEmbeddingsPath="${basePath}train.col.vecs"

devMatrix="${basePath}dev.mtx"
testMatrix="${basePath}test.mtx"

devPredictions="${basePath}dev.out"
testPredictions="${basePath}test.out"

trainOptionsString="--train-matrix=$trainMatrix --col-embeddings=$colEmbeddingsPath"
devOptionsString="--test-matrix=$devMatrix --col-embeddings=$colEmbeddingsPath --output-predictions=$devPredictions"
testOptionsString="--test-matrix=$testMatrix --col-embeddings=$colEmbeddingsPath --output-predictions=$testPredictions"

java -Xmx132G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TrainTACModel \
$trainOptionsString $sharedOptionsString

java -Xmx132G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TestTACModel \
$devOptionsString $sharedOptionsString

java -Xmx132G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TestTACModel \
$testOptionsString $sharedOptionsString
