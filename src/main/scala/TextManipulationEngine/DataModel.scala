package TextManipulationEngine

import opennlp.tools.ngram.NGramModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.StringList
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._
import scala.math.log


// This class will take in as parameters an instance of type
// TrainingData, nMin, nMax which are the lower and upper bounds,
// respectively, of the model n-gram window. This class serves
// as an implementation of our data model.

class DataModel(
                 td: TrainingData,
                 nMin: Int,
                 nMax: Int,
                 tfidf: Boolean
                 ) extends Serializable {


  // This private method will tokenize our text entries.


  private def tokenize (doc : String): Array[String] = {
    SimpleTokenizer.INSTANCE.tokenize(doc)
  }


  // This private method will help convert each text observation
  // and return a HashMap in which every token that appears in the
  // document is associated to the number of times it appears in the
  // document.

  // Map(token -> token_count)
  private def hashDoc(doc: String): Map[String, Double] = {
    val model = new NGramModel()
    model.add(new StringList(tokenize(doc): _*), nMin, nMax)
    model.iterator.map(
      x => (x.toString, model.getCount(x).toDouble)
    ).filter(e => ! td.stopWords.contains(e)).toMap
  }

  private val hashedData = td.data.map(e => hashDoc(e.text)).cache

  // Compute required idf data.

  private val numDocs: Double = hashedData.count.toDouble

  // Creates token-gram universe.
  private def createUniverse(u: RDD[Map[String, Double]]): RDD[(Long, (String, Double))] = {
    u.flatMap(identity).map(
      e => (e._1, 1.0)
    ).reduceByKey(_ + _).map(
        e => (e._1, log(numDocs / e._2))
      ).zipWithIndex.map(_.swap)
  }


  // Create token universe zipped, and corresponding idf values.
  private var universe = createUniverse(hashedData).cache
  private val numTokens = universe.count.toInt


  // Transforms a given string document into a data vector
  // based on the given data model (tfidf indicates whether
  // tfidf transformation performed.
  def transform(doc: String, tfidf: Boolean = tfidf): Vector = {

    // Some helper funcitons.
    val f : Int => String = (k : Int) => universe.lookup(k)(0)._1
    val g = (map: Map[String, Double], k: Int) => {
      val x = map.getOrElse(f(k), 0.0)
      val n = map.values.sum
      if (tfidf) (x / n) * universe.lookup(k)(0)._2 else x
    }


    val hashedDoc = hashDoc(doc)
    val indexSeq : Seq[Int] = (0 until numTokens).filter(
      k => hashedDoc.keySet.contains(f(k))
    )

    Vectors.sparse(numTokens, indexSeq.map(k => (k, g(hashedDoc, k))))
  }


  // Returns a data instance that is ready to be used for
  // model training.
  def transformData: RDD[LabeledPoint] = {

    // Some helper functions.
    val e = (x : Map[String, Double], y : (String, Double)) => {
      val n = x.values.sum
      val z = x.getOrElse(y._1, 0.0)
      if (tfidf) (z / n) * y._2 else z
    }
    val f = (obs : Observation, dataId : Long) => (dataId, (obs.label, hashDoc(obs.text)))
    val g = (x : (Long, (Double, Map[String, Double])), y : (Long, (String, Double))) =>
      ((x._1, x._2._1), (y._1.toInt, e(x._2._2, y._2)))

    // Map training data to RDD[LabeledPoint].
    td.data.zipWithUniqueId.map(
      e => f(e._1, e._2)
    ).cartesian(universe).filter(
      e => e._1._2._2.keySet.contains(e._2._2._1)).map(
        e => g(e._1, e._2)).groupByKey.map(
        e => LabeledPoint(
          e._1._2, Vectors.sparse(numTokens, e._2.toSeq))
      )

  }
}
