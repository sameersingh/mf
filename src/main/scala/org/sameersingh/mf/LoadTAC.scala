package org.sameersingh.mf

import cc.factorie.util.CmdOptions
import scala.util.Random
import java.util.zip.GZIPInputStream
import java.io._
import scala.io.Source
import org.sameersingh.mf.learner._
import cc.factorie.la.{Tensor2, DenseTensor1, Tensor}
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable

object TrainTACModel {
  import LoadTAC._
  trait TrainTACOptions extends CmdOptions with MatrixFactorizationOptions {
    val trainMatrixPath = new CmdOption("train-matrix", prefix + "train.matrix", "FILE",
      "File containing cells of train matrix, with tab-separated rowname,colname,value on each line")
    val colEmbeddingsPath = new CmdOption("col-embeddings", prefix + "col-embeddings.model", "FILE",
      "Where to write file containing embeddings for columns, with format colname\\t(value )+, (column name then tab then space-separated latent values).")
    val rowEmbeddingsPath = new CmdOption("row-embeddings", "", "FILE",
      "Where to write file containing embeddings for rows, with format rowname\\t(value )+, (row name then tab then space-separated latent values). Empty string if you don't want to write anything.")
  }
  def main(args: Array[String]): Unit = {
    val opts = new TrainTACOptions {}
    opts.parse(args)
    implicit val random = new Random(0)
    implicit val cache = new Cache
    val trainMatrix = LoadTAC.loadMatrix(opts.trainMatrixPath.value, "train")
    val fcfg = loadFactorizationConfig(opts)
    val fopt = new BasicFactorization(trainMatrix, fcfg) with LogisticLoss
//    println(fopt.avgValue(trainMatrix.trainCells))
    fopt.optimize
//    fopt.debug
//    println(fopt.avgValue(trainMatrix.trainCells))
    if (opts.colEmbeddingsPath.value != "")
      writeEmbeddings(trainMatrix.colIDs.view.map(id => (id, fopt.params.colTensor(id))), opts.colEmbeddingsPath.value)
    if (opts.rowEmbeddingsPath.value != "")
      writeEmbeddings(trainMatrix.rowIDs.view.map(id => (id, fopt.params.rowTensor(id))), opts.rowEmbeddingsPath.value)
  }
  def writeEmbeddings(embeddings: Seq[(ID, Tensor)], fileName: String): Unit = {
    write(fileName)(writer => {
      embeddings.foreach({
        case (id, t) =>
          val colStr = id.id + "\t" + t.asArray.mkString(" ")
          writer.write(colStr)
          writer.newLine()
      })
    })
  }
}

object TestTACModel {
  import LoadTAC._
  trait TestTACOptions extends CmdOptions with MatrixFactorizationOptions {
    val testMatrixPath = new CmdOption("test-matrix", prefix + "test.matrix", "FILE",
      "File containing cells of test matrix, with tab-separated rowname,colname,value on each line. Include \"?\" in cells you want to predict.")
    // TODO should I make this binary? Why not just save a hash table of tensors or somthing?
    val colEmbeddingsPath = new CmdOption("col-embeddings", prefix + "col-embeddings.model", "FILE",
      "File containing embeddings for columns, with format colname\\t(value )+, (column name then tab then space-separated latent values).")
    val outputPredictionsPath = new CmdOption("output-predictions", prefix + "univ-schema-predictions.matrix", "FILE", "File to write predictions for \"?\" cells in rowname,colname,value format.")
  }
  def main(args: Array[String]): Unit = {
    val opts = new TestTACOptions {}
    opts.parse(args)
    implicit val random = new Random(0)
    implicit val cache = new Cache
    val testMatrix = LoadTAC.loadMatrix(opts.testMatrixPath.value, "test")
    val rowIds = testMatrix.rowIDs.map(id => (id.id, id)).toMap
    val colIds = testMatrix.colIDs.map(id => (id.id, id)).toMap
    val fcfg = loadFactorizationConfig(opts).copy(updateCols = false)
    val fopt = new BasicFactorization(testMatrix, fcfg) with LogisticLoss;
    {
      val colEmbeddingsTensor = fopt.params.colFactors.value: Tensor2
      val embeddingSize = colEmbeddingsTensor.dim2
      val colEmbeddings = readEmbeddings(opts.colEmbeddingsPath.value)
      for (col <- testMatrix.colIDs; embedding <- colEmbeddings.get(col.id)) {
        val offset = col.idx * embeddingSize
        embedding.foreachActiveElement((i, v) => colEmbeddingsTensor(offset + i) = v)
      }
    }
//    println(fopt.avgValue(testMatrix.trainCells))
    fopt.optimize
//    fopt.debug
//    println(fopt.avgValue(testMatrix.trainCells))
    val preds = new ArrayBuffer[String]
    Source.fromFile(opts.testMatrixPath.value, "UTF-8").getLines().filter(_.contains("?")).foreach(line => {
      val Array(rowName, colName, _) = line.split("\t")
      // is 0.0 a good default prediction for stuff it hasn't seen?
      var p = 0.0
      for (rowid <- rowIds.get(rowName); colid <- colIds.get(colName))
        p = fopt.pred(fopt.params.rowTensor(rowIds(rowName)), fopt.params.colTensor(colIds(colName)))
      preds += (rowName + "\t" + colName + "\t" + p)
    })
    write(opts.outputPredictionsPath.value)(writer => {
      preds.foreach(line => {
        writer.write(line)
        writer.newLine()
      })
    })
  }
  def readEmbeddings(fileName: String)(implicit c: Cache): mutable.HashMap[String, Tensor] = {
    val embeddings = new mutable.HashMap[String, Tensor]
    Source.fromFile(fileName).getLines().foreach(line => {
      val Array(id, t) = line.split("\t")
      val entries = t.split(" ")
      val tensor = new DenseTensor1(entries.size)
      for (i <- 0 until entries.size) tensor += (i, entries(i).toDouble)
      embeddings(id) = tensor
    })
    embeddings
  }
}

object LoadTAC {
  trait MatrixFactorizationOptions extends CmdOptions {
    val dimensionSize = new CmdOption("embedding-size", 50, "INT", "Size of latent space for columns or rows")
    val lambda = new CmdOption("lambda", 0.1, "DOUBLE", "Strength of l2 regularization for learning")
    val iterations = new CmdOption("iterations", 10, "INT", "Number of passes over the training data for learning")
    val baseRate = new CmdOption("base-rate", 0.01, "DOUBLE", "Base learning rate for SGD")
    val logEvery = new CmdOption("log-every", 1, "INT", "Log every N iterations")
  }
  def loadFactorizationConfig(opts: MatrixFactorizationOptions): FactorizationConfig =
    FactorizationConfig(k = opts.dimensionSize.value, lambda = opts.lambda.value, baseRate = opts.baseRate.value, iterations = opts.iterations.value, logEvery = opts.logEvery.value)
  val prefix = "/home/luke/canvas/universal-schema-features/feature-matrix-factorization/runs/run20140507_features/"
  def write(fileName: String)(body: BufferedWriter => Unit): Unit = {
    val stream = new FileOutputStream(fileName)
    val writer = new BufferedWriter(new OutputStreamWriter(stream))
    body(writer)
    writer.close()
  }
  def read(fileName: String)(body: BufferedReader => Unit): Unit = {
    val stream = new FileInputStream(fileName)
    val reader = new BufferedReader(new InputStreamReader(stream))
    body(reader)
    reader.close()
  }
  def loadMatrix(filename: String, matrixName: String, gzip: Boolean = false)(implicit cache: Cache, random: Random): ObservedMatrix = {
    val stream = if (gzip) new GZIPInputStream(new FileInputStream(filename)) else new FileInputStream(filename)
    val source = Source.fromInputStream(stream, "UTF-8")
    val matrix = new Matrix(matrixName)
    for (line <- source.getLines()) {
      val split = line.split('\t')
      assert(split.length == 3)
      val Array(rowName, colName, valueStr) = split
      if (valueStr != "?") {
        val rowId = CachedID(rowName, matrixName + "_rows")
        val colId = CachedID(colName, matrixName + "_cols")
        matrix += new Cell {
          val isTrain: Boolean = true
          val value: Val = DoubleValue(valueStr.toDouble)
          val col: ID = colId
          val row: ID = rowId
          val inMatrix: ObservedMatrix = matrix
        }
      }
    }
    source.close()
    matrix
  }
}