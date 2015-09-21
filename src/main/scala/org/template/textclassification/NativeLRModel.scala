package org.template.textclassification

import java.io.Serializable

import org.apache.spark.SparkContext
import org.apache.spark.sql.{functions, UserDefinedFunction, SQLContext, DataFrame}

import scala.math._

/**
 * Created by burtn on 15/07/15.
 */
trait NativeLRModel {
  case class LREstimate (
                          coefficients : Array[Double],
                          intercept : Double
                          ) extends Serializable


  def fitLRModels:Seq[(Double, LREstimate)]

  def predict(text: String) : PredictedResult

  def prepareDataFrame(sc : SparkContext, pd : PreparedData, labels: Seq[Double]): DataFrame = {
    // 1. Import SQLContext for creating DataFrame.
    val sql: SQLContext = new SQLContext(sc)
    import sql.implicits._

    // 2. Initialize logistic regression model with regularization parameter.

    labels.foldLeft(pd.transformedData.map(x => x.point).toDF)(//transform to Spark DataFrame

      // Add the different binary columns for each label.
      (data: DataFrame, label: Double) => {
        // function: multiclass labels --> binary labels
        val f: UserDefinedFunction = functions.udf((e: Double) => if (e == label) 1.0 else 0.0)

        data.withColumn(label.toInt.toString, f(data("label")))
      }
    )
  }

  // 4. Enable vector inner product for prediction.

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    x.zip(y).map(e => e._1 * e._2).sum
  }

  // 5. Define prediction rule.
  def predict(text : String,  pd : PreparedData,lrModels:Seq[(Double, LREstimate)]): PredictedResult = {
    val x : Array[Double] = pd.transform(text).vector.toArray

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
  }
}