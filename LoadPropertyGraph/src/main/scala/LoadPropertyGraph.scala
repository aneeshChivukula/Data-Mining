import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD._
import org.apache.spark.rdd._
import org.apache.spark.graphx._
import org.apache.spark.graphx.Edge
import org.apache.spark.graphx.VertexRDD
import java.io._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.MutableList



//class Helpers {
//
//  def removemissingvalues(line: String) ={
//    val splitline = line.split(',');
//    if(splitline(0) != null)
//      print(line)    
//  }
//  
//}


object LoadPropertyGraph {
  def main(args: Array[String]) {
    
    val conf = new SparkConf().setAppName("Data Loader")
    val sc = new SparkContext(conf)
    
    val cspaperauthoraffiliations : RDD[(String, String)] =  sc.textFile("hdfs://atlas8:9000/data/cspaperauthoraffiliationspidaid.csv").map( line => line.split(",")).map(row => (row(0).toString, row(1).toString))
    val cspaperkeywords =  sc.textFile("hdfs://atlas8:9000/data/cspaperkeywordspidfidkn.csv")

    val cspapers : RDD[(String,String)] =  sc.textFile("hdfs://atlas8:9000/data/cspaperspidcsidmvn.csv").map( line => line.split(",")).map(row => (row(0).toString, row(1).toString))
    
    val conferenceinstances: RDD[(String,String)] =  sc.textFile("hdfs://atlas8:9000/data/conferenceinstancescsidcsd.csv").map( line => line.split(",")).map(row => (row(0).toString, row(1).toString))
    
    
    val cspapersauthorsconferencesjoin = cspaperauthoraffiliations.join(cspapers)
    cspapersauthorsconferencesjoin.cache()
    
    val cspapersauthorsconferencesdatesjoin = cspapersauthorsconferencesjoin.map( k =>
        (k._2._2,(k._2._1,k._1))
    ).join(conferenceinstances)
    cspapersauthorsconferencesdatesjoin.cache()
    
    val cspapersauthorsconferencesdatesjoin2015 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2015"
      )
/*
    val cspapersauthorsconferencesdatesjoin2014 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2014"
      )
      
    val cspapersauthorsconferencesdatesjoin2013 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2013"
      )

    val cspapersauthorsconferencesdatesjoin2012 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2012"
      )

    val cspapersauthorsconferencesdatesjoin2011 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2011"
      )

      
    val cspapersauthorsconferencesdatesjoin2010 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2010"
      )

    val cspapersauthorsconferencesdatesjoin2009 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2009"
      )

    val cspapersauthorsconferencesdatesjoin2008 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2008"
      )
      
    val cspapersauthorsconferencesdatesjoin2007 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2007"
      )
      
    val cspapersauthorsconferencesdatesjoin2006 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2006"
      )

    val cspapersauthorsconferencesdatesjoin2005 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2005"
      )

    val cspapersauthorsconferencesdatesjoin2004 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2004"
      )
 */     
//    cspapersauthorsconferencesdatesjoin2015.collect.foreach(println)

      
    
    
    val vertexRDDWithIndex = cspapersauthorsconferencesdatesjoin2015.map(t => t._2._1._1).distinct().zipWithIndex
    vertexRDDWithIndex.cache()
    var vertexRDDarray =  vertexRDDWithIndex.map{case (k,v) => (v,(k))}.collect()
    var vertexRDD = sc.parallelize(vertexRDDarray)
//    vertexRDD.collect.foreach(println)
    
    def makeedgelisttuples(t: (String,Iterable[String])):MutableList[(Long,Long,(String))] ={
          val it = t._2.iterator
          val l:MutableList[(Long,Long,(String))] = MutableList()
          for((a,i) <- it.zipWithIndex){
              val avidx = vertexRDDWithIndex.lookup(a)(0)
              for((b,j) <- it.zipWithIndex){
                if(j>i){
                  val bvidx = vertexRDDWithIndex.lookup(b)(0)
                  l += Tuple3(avidx,bvidx,(t._1))
                }
 
              }
            }
//          print("l.length"+l.length)
          return l
    }  
    // There is no alternative to using a Array for packing and unpacking tuples incrementally after a join
    // Is group by the best way to generate edge lists?
    
    def makeedgelistclasses(e: (Long,Long,(String))):Edge[(String)] ={
       return new Edge(e._1,e._2,(e._3))
    }
    
    var edgeListJoin = cspapersauthorsconferencesdatesjoin2015.map(t => (t._1, t._2._1._1)).groupByKey().collect()
    var edgeRDDArray = edgeListJoin.flatMap(makeedgelisttuples).map(makeedgelistclasses)
    val edgeRDD: RDD[Edge[(String)]] =  sc.parallelize(edgeRDDArray)
    
    type MyVertex = (String)
    type MyEdge = (String)    
    val CoauthorshipNetwork2015: Graph[MyVertex, MyEdge] = Graph.apply(vertexRDD,edgeRDD)
      // All vertices must be uniquely identified with 64bit longs

    
    CoauthorshipNetwork2015.vertices.collect.foreach(println)
    CoauthorshipNetwork2015.edges.collect.foreach(println)
    // GraphOps Examples
    // http://ampcamp.berkeley.edu/big-data-mini-course/graph-analytics-with-graphx.html
    // http://spark.apache.org/docs/latest/graphx-programming-guide.html#vertex_and_edge_rdds
  }
 
}
/*
Spark-submit configuration :
spark-submit --class "LoadPropertyGraph" --master yarn --deploy-mode cluster --driver-memory 16G --executor-memory 16G --num-executors 14 --executor-cores 7 ~/data-loaders_2.11-1.0.jar
ssh achivuku@atlas8
scp ./target/scala-2.11/data-loaders_2.11-1.0.jar achivuku@atlas8:~


HDFS Shell Commands :

hadoop fs -copyFromLocal cspaperauthoraffiliationspidaid.csv /data/
hadoop fs -copyFromLocal cspaperspidcsidmvn.csv /data/
hadoop fs -copyFromLocal cspaperkeywordspidfidkn.csv /data/
hadoop fs -copyFromLocal conferenceinstancescsidcsd.csv /data/

grep -Ev $',$' cspaperspidcsidmvn.csv > cspaperspidcsidmvn2.csv
grep -Ev $'\s' cspaperspidcsidmvn2.csv > cspaperspidcsidmvn3.csv
grep -Ev $'-' cspaperspidcsidmvn3.csv > cspaperspidcsidmvn4.csv
grep -Ev $'[a-z]' cspaperspidcsidmvn4.csv > cspaperspidcsidmvn5.csv
cp cspaperspidcsidmvn5.csv cspaperspidcsidmvn.csv
hadoop fs -put cspaperspidcsidmvn.csv /data/

sed 's/\[//g' conferenceinstancescsidcsd.csv > conferenceinstancescsidcsd2.csv
grep -Ev $'^,' conferenceinstancescsidcsd2.csv > conferenceinstancescsidcsd3.csv
sed 's/\]//g' conferenceinstancescsidcsd3.csv > conferenceinstancescsidcsd4.csv
grep -Ev $',$' conferenceinstancescsidcsd4.csv > conferenceinstancescsidcsd5.csv
cp conferenceinstancescsidcsd5.csv conferenceinstancescsidcsd.csv 
hadoop fs -rmr /data/conferenceinstancescsidcsd.csv
hadoop fs -copyFromLocal conferenceinstancescsidcsd.csv /data/conferenceinstancescsidcsd.csv
     
Compared to SQL joins, RDD joins are significantly faster 
Join operation is defined only on PairwiseRDDs which are quite different from a relation / table in SQL. Each element of PairwiseRDD is a Tuple2 where the first element is the key and the second is value. Both can contain complex objects as long as key provides a meaningful hashCode
http://stackoverflow.com/questions/33321704/join-two-rdd-in-spark

RDD API
http://spark.apache.org/docs/1.1.1/api/scala/index.html#org.apache.spark.rdd.RDD
SchemaRDD (Spark 1.1.1 JavaDoc)
https://spark.apache.org/docs/1.1.0/api/java/org/apache/spark/sql/SchemaRDD.html
Spark PairedRDD API
https://spark.apache.org/docs/0.6.2/api/core/spark/PairRDDFunctions.html
Scala Collections API
http://www.scala-lang.org/docu/files/collections-api/collections.html
http://www.scala-lang.org/api/2.7.6/scala/Iterator.html
GraphX API
http://spark.apache.org/docs/1.1.1/api/scala/index.html#org.apache.spark.graphx.Graph$
GraphOps Examples
http://ampcamp.berkeley.edu/big-data-mini-course/graph-analytics-with-graphx.html


Searching for date by string and regex matching. Use java.sql.Date with groupBy on pairedRDDs for fine grained date manipulation.
(start/86400 to end/86400).map(day => (day, rdd.filter(rec => rec.time 
>= day*86400 && rec.time < (day+1)*86400))) 

*/


/*
Temporary Code Snippets :

    println(edgeRDDArray(0))
    edgeRDDArray.foreach(
        a =>
        println(a._1)
        )


//    edgeRDDArray.foreach(t =>
//      t.foreach(
//          a=>
//            println(a)
//          )
//    )
    
    
    
//    edgeRDDArray.foreach( t =>
//      println(t)
//    )
    
//    edgeRDDArray.foreach(
//      t=>
//        t.foreach(
//            a=>
//      println(a)
//      for(j <- 0 until a.length){
//        println(a(j))
//      }    
//    )
//    )
//    val file = new File("/home/achivuku/output/filteredcspublications/rddoutputs/cspapersauthorsconferencesdatesjoin2015.csv")
    
//    for(i <- 0 until edgeRDDArray.length){
//    }
    
    
    // http://stackoverflow.com/questions/24497389/using-scala-to-dump-result-processed-by-spark-to-hdfs
    // http://stackoverflow.com/questions/4604237/how-to-write-to-a-file-in-scala    
    
    
//    edgeRDDArray.saveAsTextFile("file:///home/achivuku/output/filteredcspublications/rddinputs/cspapersauthorsconferencesdatesjoin2015")
    
//    val edgeRDD: RDD[(String,String,String)] =  sc.textFile("file:///home/achivuku/output/filteredcspublications/rddinputs/cspapersauthorsconferencesdatesjoin2015/").map( line => line.split(",")).map(row => (row(0).toString, row(1).toString, row(2).toString))
//    val CoauthorshipNetwork2015: Graph[String, (String,String,(String))] = Graph.apply(vertexRDD,edgeRDD, "DUMMY")
    // Array cannot be there in RHS. Only an RDD of strings
    
//    for((x,i) <- xs.view.zipWithIndex) println("String #" + i + " is " + x)

    
//    val edgeRDD =  cspapersauthorsconferencesdatesjoin2015.map(t => t._2._1._1).toArray()  
        
    
//    cspapersauthorsconferencesjoin.map(k => k._1).collect().take(1).foreach(println)
//    cspapersauthorsconferencesjoin.map(k => k._2._1).collect().take(1).foreach(println)
//    cspapersauthorsconferencesjoin.map(k => k._2._2).collect().take(1).foreach(println)
//    cspapersauthorsconferencesjoin.collect().take(1).foreach(println)
    
//    val conferenceinstancesnomissing = conferenceinstances.filter(line => !(line.split(',')(0).replaceAll("\\[","") == ""))
//    conferenceinstancesnomissing.cache()
//    println("conferenceinstances.count() " + conferenceinstances.count())
//    println("conferenceinstancesnomissing.count() " + conferenceinstancesnomissing.count())

//    val cspapersauthorsconferencesreduce = (cspaperauthoraffiliations union cspapers).reduceByKey(_ ++ _) 
//    println("cspapers.count() " + cspapers.count())
//    println("cspaperauthoraffiliations.count() " + cspaperauthoraffiliations.count())
//    println("cspapersauthorsconferencesreduce.count() " + cspapersauthorsconferencesreduce.count())
//    println("cspapers.first() " + cspapers.first())
//    cspapersauthorsconferencesjoin.collect().foreach(println)
//    cspapersauthorsconferences.map(t => sline = (t(2), t(0), t(1)))
    
//    def removemissingvalues(line: String):Boolean ={
//      return !(line.split(',')(0).replaceAll("\\[","") == "")
//    }
//    val rdd1 = conferenceinstances.filter(removemissingvalues)
//    rdd1.collect().foreach(println)

//    def makeedgelistarray(t: (String,Array[Iterable[String]])):Array[Edge[(String,String,(String))]] ={
//          val a:Array[Edge[(String,String,(String))]] = new Array[Edge[(String,String,(String))]](t._2.length * t._2.length)
//          for(i <- 0 until t._2.length){
//              for(j <- i+1 until t._2.length){
//                a:+ (t._2(i),t._2(j),(t._1))
//                
//              }
//            }
//          return a
//    }  
    
    
//    val rdd1 = conferenceinstances.flatMap(x => 
//      if(x.split(",")(0) != null)
//        x
//      else
//        None    
//    )
//    rdd1.collect().foreach(println)
//    val rdd1 = conferenceinstances.map(new Helpers().removemissingvalues())


//    edgeListJoin.take(1).foreach(
//    compactbuffer => 
//      for((a,i) <- compactbuffer._2.iterator.zipWithIndex){
//        println(a,i)
//        for((b,j) <- compactbuffer._2.iterator.zipWithIndex){
//          if(j>i){
//            println((a,b,compactbuffer._1))
//          }
//        }
//      }
////      compactbuffer._2.iterator.hasNext      
//    )

*/