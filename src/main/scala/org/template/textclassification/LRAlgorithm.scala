package org.template.textclassification

import io.prediction.controller.Params
import io.prediction.controller.P2LAlgorithm
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.UserDefinedFunction
import com.github.fommil.netlib.F2jBLAS


import scala.math._


case class LRAlgorithmParams (
  regParam  : Double
) extends Params


class LRAlgorithm(
  val sap: LRAlgorithmParams
) extends P2LAlgorithm[PreparedData, LRModel, Query, PredictedResult] {

  // Train your model.
  def train(sc: SparkContext, pd: PreparedData): LRModel = {
    new LRModel(sc, pd, sap.regParam)
  }

  // Prediction method for trained model.
  def predict(model: LRModel, query: Query): PredictedResult = {
    model.predict(query.text)
  }
}

class LRModel (
  sc : SparkContext,
  pd : PreparedData,
  regParam : Double
) extends Serializable {

  // 1. Import SQLContext for creating DataFrame.
  private val sql : SQLContext = new SQLContext(sc)
  import sql.implicits._

  // 2. Initialize logistic regression model with regularization parameter.
  private val lr = new LogisticRegression()
  .setMaxIter(100)
  .setThreshold(0.5)
  .setRegParam(regParam)

  private val data : DataFrame = pd.transformedData.toDF
  private val labels : Seq[Double] = pd.categoryMap.keys.toSeq


  private case class LREstimate (
    coefficients : Array[Double],
    intercept : Double
  ) extends Serializable



  // 3. Create a logistic regression model for each class.
  private val lrModels : Seq[(Double, LREstimate)] = labels.map(
    label => {
      val f : UserDefinedFunction = functions.udf((e : Double) => if (e == label) 1.0 else 0.0)


      val fit = lr.setLabelCol("new label").fit(
        data.withColumn(
          "new label",
          f(data("label"))
        )
      )

      // Return (label, feature coefficients, and intercept term.
      (label, LREstimate(fit.weights.toArray, fit.intercept))

    }
  )

  // 4. Enable vector inner product for prediction.
  private val f2jBLAS = new F2jBLAS

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    f2jBLAS.ddot(x.size, x, 1, y, 1)
  }

  // 5. Define prediction rule.
  def predict(text : String): PredictedResult = {
    try {
      val x : Array[Double] = pd.transform(text).toArray

    // Logistic Regression binary formula for positive probability.
    // According to MLLib documentation, class labeled 0 is used as pivot.
    // Thus, we are using:
    // log(p1/p0) = log(p1/(1 - p1)) = b0 + xTb =: z
    // p1 = exp(z) * (1 - p1)
    // p1 * (1 + exp(z)) = exp(z)
    // p1 = exp(z)/(1 + exp(z))
      val pred = lrModels.map(
        e => {
          val z = exp(innerProduct(e._2.coefficients, x) + e._2.intercept)
            (e._1, z / (1 + z))
        }
      ).maxBy(_._2)

      PredictedResult(pd.categoryMap(pred._1), pred._2)
    } catch {
      case e : IllegalArgumentException => PredictedResult(pd.majorityCategory, 0)
    }
  }
}


