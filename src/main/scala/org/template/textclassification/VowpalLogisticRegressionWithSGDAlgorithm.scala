package org.template.textclassification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import grizzled.slf4j.Logger

import java.nio.file.{Files, Paths}

import vw.VW

case class AlgorithmParams(
  maxIter: Int,
  regParam: Double,
  stepSize: Double,
  bitPrecision: Int,
  modelName: String,
  namespace: String,
  ngram: Int
) extends Params

// extends P2LAlgorithm because VW doesn't contain RDD.
class VowpalLogisticRegressionWithSGDAlgorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Array[Byte], Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Array[Byte] = {
   
    require(!data.td.data.take(1).isEmpty,
      s"RDD[labeldPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preprator generates PreparedData correctly.")
  
    val reg = "--l2 " + ap.regParam
    //val iters = "-c -k --passes " + ap.maxIter
    val lrate = "-l " + ap.stepSize
    val ngram = "--ngram " + ap.ngram 
  
    val vw = new VW("--loss_function logistic --invert_hash readable.model -b " + ap.bitPrecision + " " + "-f " + ap.modelName + " " + reg + " " + lrate + " " + ngram)
    
    val inputs = for (point <- data.transformedData.collect) yield (if (point.point.label.toDouble == 0.0) "-1.0" else "1.0") + " |" + ap.namespace + " " + rawTextToVWFormattedString(point.text) + " "  + vectorToVWFormattedString(point.point.features)

    //val inputs = for (point <- data.transformedData) yield (if (point.label.toDouble == 0.0) "-1.0" else "1.0") + " |" + ap.namespace + " "  + rawTextToVWFormattedString(point.)

     //Regressing    
    //val inputs = for (point <- data.td.data) yield point.category.toDouble.toString + " |" + ap.namespace + " "  + rawTextToVWFormattedString(point.text)


    //for (item <- inputsCollected) logger.info(item)

    val results = for (item <- inputs) yield vw.learn(item)

    val matchOnTrainSet = for (item <- inputs) yield  item.startsWith(if(vw.predict(item).toDouble  > 0.5) "1" else "-1")


    val acc = (for (x <- matchOnTrainSet) yield if(x) 1 else 0).sum.toDouble / matchOnTrainSet.size
    println("Accuracy on Training set: " + acc)

    vw.close()
     
    Files.readAllBytes(Paths.get(ap.modelName))
  }

  def predict(byteArray: Array[Byte], query: Query): PredictedResult = {
    Files.write(Paths.get(ap.modelName), byteArray)

    val vw = new VW("--link logistic -i " + ap.modelName)
    val pred = vw.predict("|" + ap.namespace + " " + rawTextToVWFormattedString(query.text)).toDouble 
    vw.close()

    val category = (if(pred > 0.5) 1 else 0).toString
    val prob = (if(pred > 0.5) pred else 1.0 - pred)
    val result = new PredictedResult(category, prob)
   
    result
  }

  def rawTextToVWFormattedString(str: String) : String = {
     //VW input cannot contain these characters 
     str.replaceAll("[|:]", " ")
  }

  def vectorToVWFormattedString(vec: Vector): String = {
     vec.toArray.zipWithIndex.map{ case (dbl, int) => s"$int:$dbl"} mkString " "
  }

}
