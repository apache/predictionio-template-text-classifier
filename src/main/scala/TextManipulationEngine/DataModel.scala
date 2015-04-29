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
                 nMax: Int
                 ) extends Serializable {


  // This private method will tokenize our text entries.

  // document => token list
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
    val map  = model.iterator.map(
      x => (x.toString, model.getCount(x).toDouble)
    ).filter(e => ! td.stopWords.contains(e)).toMap
    map.mapValues(e => e / map.values.sum)
  }

  private val hashedData = td.data.map(e => hashDoc(e.text)).cache

  // Compute required idf data.

  private val numDocs: Double = hashedData.count.toDouble

  // Creates token-gram universe.
  private def createUniverse(u: RDD[Map[String, Double]]): RDD[((String, Double), Long)] = {
    u.flatMap(identity).map(
      e => (e._1, 1.0)
    ).reduceByKey(_ + _).map(
        e => (e._1, log(numDocs / e._2))
      ).zipWithIndex
  }
  private val universe = createUniverse(hashedData)

  // Map(token -> idf)
  private val idf : Map[String, Double] = universe.map(_._1).collect.toMap
  // Map(token -> token index in universe)
  private val tokenIndex : Map[String, Int] = universe.map(e => (e._1._1, e._2.toInt)).collect.toMap
  private val numTokens = idf.size


  // Transforms a given string document into a data vector
  // based on the given data model (tfidf indicates whether
  // tfidf transformation performed.
  def transform(doc: String): Vector = {
    val hashedDoc = hashDoc(doc)
    Vectors.sparse(numTokens, hashedDoc.map(e => (tokenIndex.get(e._1).get,
      e._2 * idf.get(e._1).get)).toArray)
  }


  // Returns a data instance that is ready to be used for
  // model training.
  def transformData: RDD[LabeledPoint] = {

    td.data.map(obs => LabeledPoint(
        obs.label, transform(obs.text)
      ))
    }

}
