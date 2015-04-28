package TextManipulationEngine

import opennlp.tools.ngram.NGramModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.StringList
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
                 val td: TrainingData,
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

  private val hashedData = td.data.map(e => hashDoc(e.text))

  // Compute required idf data.

  private val numDocs: Double = hashedData.count().toDouble

  // Creates token-gram universe.
  private def createUniverse(u: RDD[Map[String, Double]]): Array[(String, Double)] = {
    u.flatMap(identity).map(
      e => (e._1, 1.0)
    ).reduceByKey(_ + _).map(
        e => (e._1, log(numDocs / e._2))
      ).collect()
  }


  // Create token universe zipped, and corresponding idf values.
  val universe = createUniverse(hashedData)



  // Transforms a given string document into a data vector
  // based on the given data model (tfidf indicates whether
  // tfidf transformation performed.
  def transform(doc: String, tfidf: Boolean = tfidf): Array[Double] = {
    val hashedDoc = hashDoc(doc)
    val n = hashedDoc.values.sum
    val x = universe.map(e => hashedDoc.getOrElse(e._1, 0.0) / n)
    if (tfidf)
      x.zip(universe.map(_._2)).map(e => e._1 * e._2)
    else x
  }


  // Returns a data instance that is ready to be used for
  // model training.
  def transformData: RDD[LabeledPoint] = {
    td.data.map(
      e => LabeledPoint(e.label, Vectors.dense(transform(e.text)))
    )
  }

}
