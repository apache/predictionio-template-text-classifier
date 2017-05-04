name := "org.example.textclassification"

libraryDependencies ++= Seq(
  "org.apache.predictionio" %% "apache-predictionio-core" % "0.11.0-incubating" % "provided",
  "org.apache.spark"        %% "spark-core"               % "1.4.1" % "provided",
  "org.apache.spark"        %% "spark-mllib"              % "1.4.1" % "provided",
  "org.apache.lucene"        % "lucene-core"              % "6.5.1")
