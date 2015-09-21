
name := "org.template.textclassification"

organization := "io.prediction"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "io.prediction"    % "core_2.10"        % pioVersion.value % "provided",
  "org.apache.spark" %% "spark-core" % "1.4.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.4.1" % "provided",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
  "com.github.johnlangford" % "vw-jni" % "8.0.0",
  "org.xerial.snappy" % "snappy-java" % "1.1.1.7"
)

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
  {
    case y if y.startsWith("doc")     => MergeStrategy.discard
    case x => old(x)
  }
}
