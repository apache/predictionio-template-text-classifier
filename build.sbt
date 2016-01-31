name := "org.template.textclassification"

organization := "io.prediction"

scalaVersion := "2.10.5"

organization := "io.prediction"

libraryDependencies ++= Seq(
  "io.prediction"    %% "core"     % pioVersion.value % "provided",
  "org.apache.spark" %% "spark-core"    % "1.4.1" % "provided",
  "org.apache.spark" %% "spark-mllib"   % "1.4.1" % "provided")
