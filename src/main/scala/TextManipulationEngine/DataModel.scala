package TextManipulationEngine

import opennlp.tools.ngram.NGramModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.StringList
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import scala.collection.JavaConversions._
import scala.collection.immutable.HashMap
import scala.math.log





// This class will take in as parameters an instance of type
// TrainingData, nMin, nMax which are the lower and upper bounds,
// respectively, of the model n-gram window.


class DataModel (
                  val td: TrainingData,
                  val nMin: Int,
                  val nMax: Int
                  ) extends Serializable {


  // 1. Tokenizer: document => token list.
  // Takes an individual document and converts it to
  // a list of allowable tokens.

  private def tokenize (doc : String): Array[String] = {
    SimpleTokenizer.INSTANCE
      .tokenize(doc)
      .filter(e => ! td.stopWords.contains(e))
  }


  // 2. Hasher: Array[tokens] => Map(n-gram -> n-gram document tf).

  private def hash (tokenList : Array[String]): Map[String, Double] = {
    // Initialize an NGramModel from OpenNLP tools library,
    // and add the list of allowable tokens to the n-gram model.
    val model : NGramModel = new NGramModel()
    model.add(new StringList(tokenList: _*), nMin, nMax)

    val map : Map[String, Double] = model.iterator
      .map(
        x => (x.toString, model.getCount(x).toDouble)
      ).toMap

    // Divide by the total number of n-grams in the document
    // to obtain n-gram frequency.
    map.mapValues(e => e / map.values.sum)
  }


  // 3. Bigram universe extractor: RDD[bigram hashmap] => RDD[((n-gram, n-gram idf), global index)]

  private def createUniverse(u: RDD[Map[String, Double]]): RDD[((String, Double), Long)] = {
    // Total number of documents (should be 11314).
    val numDocs: Double = td.data.count.toDouble
    u.flatMap(identity)
      .map(e => (e._1, 1.0))
      .reduceByKey(_ + _)
      .map(e => (e._1, log(numDocs / e._2)))
      .zipWithIndex
  }


  // 4. Set private class variables for use in data transformations.

  // Create our n-gram universe.
  private val universe : RDD[((String, Double), Long)]= createUniverse(
    td.data
      .map(e => hash(tokenize(e.text)))
  ).cache

  // Get total number n-grams in universe (in
  // bigram case, the number of unique n-grams
  // is about 1,200,000).
  private val numTokens : Int = universe.count.toInt

  // Create ngram to idf hashmap:
  //    Map(n-gram -> n-gram idf)
  private val idf : HashMap[String, Double] = HashMap(universe.map(_._1).collect : _*)

  // Create n-gram to global index hashmap:
  //    Map(n-gram -> global index)
  private val globalIndex : HashMap[String, Int] = HashMap(universe.map(
    e => (e._1._1, e._2.toInt)
  ).collect : _*)

  // 5. Document Transformer: document => sparse tf-idf vector.
  // This takes a single document, tokenizes it, hashes it,
  // and finally returns a sparse vector containing the
  // tf-idf entries of the document n-grams (0 for all n-grams
  // not contained in the document).

  def transform(doc: String): Vector = {
    // Map(n-gram -> document tf)
    val hashedDoc = hash(tokenize(doc)).filter(e => idf.keySet.contains(e._1))
    Vectors.sparse(
      numTokens,
      hashedDoc.map {
        case (ngram, tf) => (globalIndex(ngram), idf(ngram) * tf)
      }.toArray
    )
  }


  // 6. Data Transformer: RDD[documents] => RDD[LabeledPoints]

  def transformData: RDD[LabeledPoint] = {
    td.data.map(e => LabeledPoint(e.label, transform(e.text)))
  }

}





