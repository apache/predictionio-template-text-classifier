package TextManipulationEngine

import opennlp.tools.ngram.NGramModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.StringList
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
                 tfidf: Boolean,
                 stopWords: Set[String] = Set()
                 ) extends Serializable {


  // This private method will tokenize our text entries.


  private def tokenize (doc : String): Array[String] = {
    SimpleTokenizer.INSTANCE.tokenize(doc)
  }


  // This private method will help convert each text observation
  // and return a HashMap in which every token that appears in the
  // document is associated to the number of times it appears in the
  // document.
  private def hashDoc(doc: String): Map[String, Double] = {
    val model = new NGramModel()
    model.add(new StringList(tokenize(doc): _*), nMin, nMax)
    model.iterator.map(
      x => (x.toString, model.getCount(x).toDouble)
    ).toMap.filter(e => !stopWords.contains(e._1))
  }

  private val hashedData = td.data.map(e => hashDoc(e.text))


  // Create token-gram universe.
  private def createUniverse(u: RDD[Map[String, Double]]): Array[String] = {
    u.flatMap(e => e.keySet).distinct.collect
  }

  private val universe = createUniverse(hashedData)


  // Compute required idf data.

  private val numDocs = td.data.count.toDouble

  private def computeIdf(s: String): Double = {
    log(numDocs / hashedData.filter(e => e.keySet.contains(s)).count.toDouble)
  }

  private val idfVector = universe.map(e => computeIdf(e))



  // Transforms a given string document into a data vector
  // based on the given data model (tfidf indicates whether
  // tfidf transformation performed.
  def transform(doc: String, tfidf: Boolean = tfidf): Array[Double] = {
    val hashedDoc = hashDoc(doc)
    val N = hashedDoc.values.sum
    val x = universe.map(e => hashedDoc.getOrElse(e, 0.0) / N)
    if (tfidf)
      x.zip(idfVector).map(e => e._1 * e._2)
    else x
  }


  // Returns a data instance that is ready to be used for
  // model training.
  def transformData: RDD[(Double, Array[Double])] = {
    td.data.map(
      e => (e.label, transform(e.text))
    )
  }

}
