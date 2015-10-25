package org.template.textclassification

import grizzled.slf4j.Logger
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.Params
import io.prediction.controller.PDataSource
import io.prediction.controller.SanityCheck
import io.prediction.data.store.PEventStore
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD



// 1. Initialize your Data Source parameters. This is
// where you specify your application name, MyTextApp,
// and the number of folds that are to be used for
// cross validation.

case class DataSourceParams(
                             appName: String,
                             evalK: Option[Int]
                             ) extends Params



// 2. Define your DataSource component. Remember, you must
// implement a readTraining method, and, optionally, a
// readEval method.

class DataSource (
                   val dsp : DataSourceParams
                   ) extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, ActualResult] {

  @transient lazy val logger = Logger[this.type]

  // Helper function used to store data given
  // a SparkContext.
  private def readEventData(sc: SparkContext) : RDD[Observation] = {
    //Get RDD of Events.
    PEventStore.find(
      appName = dsp.appName,
      entityType = Some("user"), // specify data entity type
      eventNames = Some(List("$set")) // specify data event name

      // Convert collected RDD of events to and RDD of Observation
      // objects.
    )(sc).map(e => {
      val label : String = e.properties.get[String]("label")
      Observation(
        if (label == "1") 1.0 else 0.0,
        e.properties.get[String]("text"),
        label
      )
    }).cache
  }

  // Helper function used to store stop words from
  // event server.
  private def readStopWords(sc : SparkContext) : Set[String] = {
    PEventStore.find(
      appName = dsp.appName,
      entityType = Some("resource"),
      eventNames = Some(List("stopwords"))

      //Convert collected RDD of strings to a string set.
    )(sc)
      .map(e => e.properties.get[String]("word"))
      .collect
      .toSet
  }


  // Read in data and stop words from event server
  // and store them in a TrainingData instance.
  override
  def readTraining(sc: SparkContext): TrainingData = {
    new TrainingData(readEventData(sc), readStopWords(sc))
  }

  // Used for evaluation: reads in event data and creates
  // cross-validation folds.
  override
  def readEval(sc: SparkContext):
  Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, ActualResult)])] = {
    // Zip your RDD of events read from the server with indices
    // for the purposes of creating our folds.
    val data = readEventData(sc).zipWithIndex()
    // Create cross validation folds by partitioning indices
    // based on their index value modulo the number of folds.
    (0 until dsp.evalK.get).map { k =>
      // Prepare training data for fold.
      val train = new TrainingData(
        data.filter(_._2 % dsp.evalK.get != k).map(_._1),
        readStopWords
          ((sc)))

      // Prepare test data for fold.
      val test = data.filter(_._2 % dsp.evalK.get == k)
        .map(_._1)
        .map(e => (new Query(e.text), new ActualResult(e.category)))

      (train, new EmptyEvaluationInfo, test)
    }
  }
}


// 3. Observation class serving as a wrapper for both our
// data's class label and document string.
case class Observation(
                        label : Double,
                        text : String,
                        category :String
                        ) extends Serializable

// 4. TrainingData class serving as a wrapper for all
// read in from the Event Server.
class TrainingData(
                    val data : RDD[Observation],
                    val stopWords : Set[String]
                    ) extends Serializable with SanityCheck {

  // Sanity check to make sure your data is being fed in correctly.

  def sanityCheck {
    try {
      val obs : Array[Double] = data.takeSample(false, 5).map(_.label)

      println()
      (0 until 5).foreach(
        k => println("Observation " + (k + 1) +" label: " + obs(k))
      )
      println()
    } catch {
      case (e : ArrayIndexOutOfBoundsException) => {
        println()
        println("Data set is empty, make sure event fields match imported data.")
        println()
      }
    }

  }

}
