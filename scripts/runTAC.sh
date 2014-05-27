basePath="/iesl/canvas/luke/universal-schema-features/feature-matrix-factorization/runs/run20140507_features/"

dimensionSize="50"
lambda="0.1"
iterations="10"
baseRate="0.01"
logEvery="1"

sharedOptionsString="--embedding-size=$dimensionSize --lambda=$lambda --iterations=$iterations --base-rate=$baseRate --log-every=$logEvery"

trainMatrix="${basePath}data/train_matrix.subset"
colEmbeddingsPath="${basePath}data/colEmbeddings.subset"
testMatrix="${basePath}data/test_matrix.subset"
outputPredictions="${basePath}data/output-predictions.subset"

trainOptionsString="--train-matrix=$trainMatrix --col-embeddings=$colEmbeddingsPath"
testOptionsString="--test-matrix=$testMatrix --col-embeddings=$colEmbeddingsPath"

java -Xmx32G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TrainTACModel \
$trainOptionsString $sharedOptionsString

java -Xmx32G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TestTACModel \
$testOptionsString $sharedOptionsString