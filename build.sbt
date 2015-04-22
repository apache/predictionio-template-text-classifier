name := "TextManipulationEngine"

libraryDependencies ++= Seq(
  "io.prediction"    %% "core"        % pioVersion.value % "provided",
  "org.apache.spark" %% "spark-core" % "1.3.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.3.1" % "provided"
)

libraryDependencies += "org.apache.opennlp" % "opennlp-tools" % "1.5.3"

libraryDependencies += "org.scalanlp" % "breeze" % "0.11.2"
