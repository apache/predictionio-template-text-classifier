package TextManipulationEngine


import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel

import scala.math.exp

class SupervisedModel(
                       val pd: PreparedData,
                       lambda: Double
                       ) extends Serializable {


  private val nb : NaiveBayesModel = NaiveBayes.train(
    pd.dataModel.transformData, lambda)

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    require(x.length == y.length)
    x.zip(y).map(e => e._1 * e._2).sum
  }


  private def getScores(doc: String): Array[Double] = {
    def normalize(u: Array[Double]): Array[Double] = u.map(_ / u.sum)
    val x: Array[Double] = pd.dataModel.transform(doc).toArray
    normalize((nb.pi zip nb.theta).map(
      e => exp(innerProduct(e._2, x) + e._1)
    ))
  }

  def predict(doc : String) : PredictedResult = {
    val x: Array[Double] = getScores(doc)
    val y: (Double, Double) = (nb.labels zip x).maxBy(_._2)
    new PredictedResult(y._1, y._2)
  }
}
