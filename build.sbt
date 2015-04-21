name := "TextManipulationEngine"

libraryDependencies ++= Seq(
  "io.prediction"    %% "core"        % pioVersion.value % "provided",
  "org.apache.spark" %% "spark-core"  % "1.2.0"          % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.2.0"          % "provided"
)

libraryDependencies += "org.apache.opennlp" % "opennlp-tools" % "1.5.3"