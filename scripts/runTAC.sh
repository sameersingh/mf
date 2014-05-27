dimensionSize="50"
lambda="0.1"
iterations="10"
baseRate="0.01"
logEvery="1"

sharedOptionsString="--embedding-size=$dimensionSize --lambda=$lambda --iterations=$iterations --base-rate=$baseRate --log-every=$logEvery"

trainMatrix="data/trainMatrix.subset"
colEmbeddingsPath="data/colEmbeddings.subset"
testMatrix="data/testMatrix.subset"
outputPredictions="data/output-predictions.subset"

trainOptionsString="--train-matrix=$trainMatrix --col-embeddings=$colEmbeddingsPath"
testOptionsString="--test-matrix=$testMatrix --col-embeddings=$colEmbeddingsPath"

java -Xmx32G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TrainTACModel \
"$trainOptionsString $sharedOptionsString"

java -Xmx32G -cp target/mf-1.0-SNAPSHOT-jar-with-dependencies.jar \
org.sameersingh.mf.TrainTACModel \
"$testOptionsString $sharedOptionsString"