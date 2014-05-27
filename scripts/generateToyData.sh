basePath="/iesl/canvas/luke/universal-schema-features/feature-matrix-factorization/runs/run20140507_features/"
#basePath=""

# always use relation feats at train and not at test
use_intertext=true
use_ngrams=false

subsetSize=1000

shuffle() {
	awk 'BEGIN{srand(0)}{printf "%06d %s\n", rand()*1000000, $0;}' | sort -n | cut -c8-
}

export -f shuffle

getCells() {
	shuffle | head -n $subsetSize | awk -F$'\t' 'BEGIN{OFS="\t"}{split($NF,a," "); for (e in a) if (a[e]!="+1") {split(a[e],s,":"); print NR,s[1],s[2]}}';
}

export -f getCells

trainRelations=$(cat "${basePath}train_relation.feats" | getCells)
testRelations=$(cat "${basePath}test_relation.feats" | getCells)
markedTestRelations=$(printf "%s\n" "$testRelations" | awk -F$'\t' 'BEGIN{OFS="\t"}{print $1,$2,"?"}')

trainOutputText=""
testOutputText=""
if [ $use_intertext = true ]; then
	echo "using intertext"
	trainIntertext=$(cat "${basePath}train_intertext.feats" | getCells)
	testIntertext=$(cat "${basePath}test_intertext.feats" | getCells)
	trainOutputText=$(printf "%s\n%s" "$trainOutputText" "$trainIntertext")
	testOutputText=$(printf "%s\n%s" "$testOutputText" "$testIntertext")
fi
if [ $use_ngrams = true ]; then
	echo "using ngrams"
	trainNGrams=$(cat "${basePath}train_ngram.feats" | getCells)
	testNGrams=$(cat "${basePath}test_ngram.feats" | getCells)
	trainOutputText=$(printf "%s\n%s" "$trainOutputText" "$trainNGrams")
	testOutputText=$(printf "%s\n%s" "$testOutputText" "$testNGrams")
fi
trainOutputText=$(printf "%s\n%s" "$trainOutputText" "$trainRelations")
testOutputText=$(printf "%s\n%s" "$testOutputText" "$markedTestRelations")
printf "%s" "$trainOutputText" | grep -v "^$" > "${basePath}data/trainMatrix.subset"
printf "%s" "$testOutputText" | grep -v "^$" > "${basePath}data/testMatrix.subset"
