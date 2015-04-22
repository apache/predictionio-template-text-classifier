package TextManipulationEngine


import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.math.exp


abstract class Model extends Serializable {
  def predict(doc: String): PredictedResult
}


class SupervisedModel(
                       val pd: PreparedData,
                       lambda: Double
                       ) extends Model {


  private val nb : NaiveBayesModel = NaiveBayes.train(
    pd.dataModel.transformData.map(
      e => LabeledPoint(e._1, Vectors.dense(e._2))
    ), lambda = lambda)

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {

    require(x.length == y.length)

   (0 until x.length).toArray.map(
        i => x(i) * y(i)
   ).sum
  }

  private def getScores(doc : String) : Array[(Double, Double)] = {
    val x: Array[Double] = pd.dataModel.transform(doc)
    (0 until nb.theta.length).toArray.map(
      i => (i.toDouble, exp(
        innerProduct(nb.theta(i), x) + nb.pi(i)
      )))
  }

  def predict(doc : String) : PredictedResult = {
    val x : Array[(Double, Double)] = getScores(doc)
    val C : Double = x.reduceLeft((a, b) => (1, a._2 + b._2))._2

    val y : (Double, Double) = x.reduceLeft((a, b) => if (a._2 >= b._2) a else b)

    new PredictedResult(y._1, y._2 / C)
  }

}

class UnsupervisedModel(pd: PreparedData, nCluster: Int) extends Model {
  def predict(doc: String): PredictedResult = {
    new PredictedResult(1, 1)
  }
}


