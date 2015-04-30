package TextManipulationEngine


import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel

import scala.math.exp

class SupervisedModel(
                       val pd: PreparedData,
                       lambda: Double
                       ) extends Serializable {



  // 1. Fit a Naive Bayes model using the prepared data.

  private val nb : NaiveBayesModel = NaiveBayes.train(
    pd.dataModel.transformData, lambda)



  // 2. Set up framework for performing the required Matrix
  // Multiplication for the prediction rule explained in the
  // tutorial.

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    require(x.length == y.length)

    x.zip(y).map(e => e._1 * e._2).sum
  }



  // 3. Given a document string, return a vector of corresponding
  // class membership probabilities.

  private def getScores(doc: String): Array[Double] = {
    // Helper function used to normalize probability scores.
    // Returns an object of type Array[Double]
    val normalize = (u: Array[Double]) => u.map(_ / u.sum)
    // Vectorize query,
    val x: Array[Double] = pd.dataModel.transform(doc).toArray

    normalize(
      nb.pi
        .zip(nb.theta)
        .map(e => exp(innerProduct(e._2, x) + e._1))
    )
  }



  // 4. Implement predict method for our model using
  // the prediction rule given in tutorial.

  def predict(doc : String) : PredictedResult = {
    val x: Array[Double] = getScores(doc)
    val y: (Double, Double) = (nb.labels zip x).maxBy(_._2)
    new PredictedResult(y._1, y._2)
  }
}
