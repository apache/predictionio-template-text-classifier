package TextManipulationEngine


import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.math.exp

class SupervisedModel(
                       val pd: PreparedData,
                       lambda: Double
                       ) extends Serializable {


  private val nb : NaiveBayesModel = NaiveBayes.train(
    pd.dataModel.transformData.map(
      e => LabeledPoint(e._1, Vectors.dense(e._2))
    ), lambda = lambda)

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    require(x.length == y.length)
    x.zip(y).map(e => e._1 * e._2).sum
  }

  private def getScores(doc : String) : Array[(Double, Double)] = {
    val x: Array[Double] = pd.dataModel.transform(doc)
    (0 until nb.pi.length).toArray.map(
      i => (i.toDouble, exp(
        innerProduct(nb.theta(i), x) + nb.pi(i)
      )))
  }

  def predict(doc : String) : PredictedResult = {
    val x : Array[(Double, Double)] = getScores(doc)
    val C = x.map(e => e._2).sum
    val y : (Double, Double) = x.reduceLeft((a, b) => if (a._2 >= b._2) a else b)

    new PredictedResult(y._1, y._2 / C)
  }

}
