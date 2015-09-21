package org.template.textclassification


import io.prediction.controller.PPreparator
import io.prediction.controller.Params
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.{IDF, IDFModel, HashingTF}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.collection.Map
import scala.collection.immutable.HashMap
import scala.collection.JavaConversions._
import scala.math._


// 1. Initialize Preparator parameters. Recall that for our data
// representation we are only required to input the n-gram window
// components.

case class PreparatorParams(
  nGram: Int,
  numFeatures: Int = 5000,
  SPPMI: Boolean
) extends Params

case class VectorAndTextExample(
                        vector: SparseVector,
                        text : String
                        ) extends Serializable

case class LabeledPointAndTextExample(
                                 point: LabeledPoint,
                                 text : String
                                 ) extends Serializable


// 2. Initialize your Preparator class.

class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {

  // Prepare your training data.
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {
    new PreparedData(td, pp.nGram, pp.numFeatures, pp.SPPMI, sc)
  }
}

//------PreparedData------------------------

class PreparedData(
  val td: TrainingData,
  val nGram: Int,
  val numFeatures: Int,
  val SPPMI: Boolean,
  @transient val sc: SparkContext
) extends Serializable {

  // 1. Hashing function: Text -> term frequency vector.

  private val hasher = new HashingTF(numFeatures = numFeatures)


  def transform(text: String): VectorAndTextExample ={
    return if(SPPMI) transformSPPMI(text) else transformTFIDF(text)
  }

  val idf : IDFModel = new IDF().fit(td.data.map(e => hashTF(e.text)))


  //3. Document Transformer: text => tf-idf vector.

  private def transformTFIDF(text : String): VectorAndTextExample = {
    // Map(n-gram -> document tf)
    val result = VectorAndTextExample(idf.transform(hashTF(text)).toSparse, text)
    //println(result)
    result
  }

  val ppmiMap = generateSPPMIMatrix(td,sc).collectAsMap()
  println(ppmiMap.head._2.size)
  println(ppmiMap.head)


  private def hashTF(text: String): Vector = {
    val newList: Array[String] = text.split(" ")
      .sliding(nGram)
      .map(_.mkString)
      .toArray

    hasher.transform(newList)
  }

  private def transformSPPMI(text : String): VectorAndTextExample = {
    // Map(n-gram -> document tf)

    val result = VectorAndTextExample(ppmiMap(text), text)
    //println(result)
    result
  }


  private def calculateSPPMI(localMat: Matrix, N: Long, k: Int): IndexedSeq[MatrixEntry] = {
    //println(localMat)
    val pmiMatrixEntries = for (i <- 0 until localMat.numCols; j <- 0 until localMat.numRows)
      yield {
        new MatrixEntry(j, i, math.max(0, math.log(localMat(j, i) * N / (localMat(i, i) * localMat(j, j))) / math.log(2.0) - math.log(k) / math.log(2.0)))
      }
    return pmiMatrixEntries
  }

  private def generateSPPMIMatrix(trainData: TrainingData, sc:SparkContext) : RDD[(String,SparseVector)] = {
    val (hashedFeats: RDD[Vector], mat: IndexedRowMatrix, cooccurrences: Matrix) = computeCooccurrences(trainData)

    val k = 10
    val pmiEntries = calculateSPPMI(cooccurrences , mat.numRows, k)
    val pmiMat: CoordinateMatrix = new CoordinateMatrix(sc.parallelize(pmiEntries))
    val indexedPMIMat = pmiMat.toIndexedRowMatrix()

    //val principalComponents = indexedPMIMat.toRowMatrix().computePrincipalComponents(500)
    //val pcPMImat = indexedPMIMat.multiply(principalComponents)

    println(trainData.data.count())
    println(indexedPMIMat.numCols())
//    println(pcPMImat.numCols())

    val pmiMatRows = indexedPMIMat.rows.map(e=> e.index -> e.vector).collectAsMap()

    return generateTextToSPPMIVectorMap(trainData, hashedFeats, pmiMatRows)
  }
  private def generateTextToSPPMIVectorMap(trainData: TrainingData, hashedFeats: RDD[Vector], pmiMatRows: Map[Long, Vector]): RDD[(String, SparseVector)] = {
    //TODO: take into account feature counts, currently it's on/off
    //also not use var
    val composedWordVectors = for (v <- hashedFeats)
      yield {
        var ar = Array.fill[Double](pmiMatRows.head._2.size)(0)
        for (i <- 0 until v.size; if v(i) > 0) {
          //Additive
          //ar = (ar,pmiMatRows(i).toArray).zipped.map(_ + _)

          //Appending
          ar = ar ++ pmiMatRows(i).toArray
        }

        Vectors.dense(ar.map(x => x)).toSparse
      }

    val textToSPPMIVectorMap = (trainData.data.map(x => x.text) zip composedWordVectors)
    textToSPPMIVectorMap
  }

  private def computeCooccurrences(trainData: TrainingData): (RDD[Vector], IndexedRowMatrix, Matrix) = {
    val hashedFeats = trainData.data.map(e => hashTF(e.text))

    val rows = hashedFeats.map( x => 
      x.toArray.map( value => if (value > 0) 1.0 else 0.0)).map( y => Vectors.dense(y).toSparse)

    val indexedRows = rows.zipWithIndex.map(x => new IndexedRow(x._2, x._1))

    val mat = new IndexedRowMatrix(indexedRows)


    //println(mat.toBlockMatrix().toLocalMatrix())

    //println(blockMat.numCols())
    //println(blockMat.numRows())

    val cooccurrences = mat.computeGramianMatrix()
    //Alternatively:
    //val cooccurrences = blockMat.transpose.multiply(blockMat)
    (hashedFeats, mat, cooccurrences)
  }








  // 4. Data Transformer: RDD[documents] => RDD[LabeledPoints]

  val transformedData: RDD[LabeledPointAndTextExample] = {
    td.data.map(e =>  LabeledPointAndTextExample(LabeledPoint(e.label, transform(e.text).vector), e.text))
  }


  // 5. Finally extract category map, associating label to category.
  val categoryMap = td.data.map(e => (e.label, e.category)).collectAsMap


}




