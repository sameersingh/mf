basePath="/iesl/canvas/luke/universal-schema-features/feature-matrix-factorization/runs/run20140507_features/"

dimensionSize="50"
lambda="0.1"
iterations="10"
baseRate="0.01"
logEvery="10000"

sharedOptionsString="--embedding-size=$dimensionSize --lambda=$lambda --iterations=$iterations --base-rate=$baseRate --log-every=$logEvery"

trainMatrix="${basePath}data/train_matrix.full"
colEmbeddingsPath="${basePath}data/colEmbeddings.full"
testMatrix="${basePath}data/test_matrix.full"
outputPredictions="${basePath}data/output-predictions.full"

trainOptionsString="--train-matrix=$trainMatrix --col-embeddings=$colEmbeddingsPath"
testOptionsString="--test-matrix=$testMatrix --col-embeddings=$colEmbeddingsPath"

java -Xmx80G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TrainTACModel \
$trainOptionsString $sharedOptionsString

java -Xmx80G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TestTACModel \
$testOptionsString $sharedOptionsString
