//spark shell --driver-memory 4g

////////////////////////////
////// 3.3 데이터 세팅 //////
////////////////////////////
//// 01 ////
val rawUserArtistData=spark.read.textFile("./Chap3_data/profiledata_06-May-2005/user_artist_data.txt")
rawUserArtistData.take(5).foreach(println)

val userArtistDF=rawUserArtistData.map{line=> 
    val Array (user, artist,_*)=line.split(' ')
    (user.toInt,artist.toInt)
}.toDF("user","artist")

userArtistDF.agg(
    min("user"),max("user"),min("artist"),max("artist")
).show()

//// 02 ////
val rawArtistData=spark.read.textFile("./Chap3_data/profiledata_06-May-2005/artist_data.txt")

rawArtistData.map{line=>
    val (id, name) = line.span(_ != '\t')
    (id.toInt, name.trim)    
}.count
//오류 발생 : .count에서 오류발생 (일부 행 제대로 파싱 안됨.)
//일부 행이 탭을 포함하지 않거나 의도하지 않은 개행문자가 포함됨.

val artistByID=rawArtistData.flatMap{ line =>
    val (id, name) = line.span(_ != '\t')
    if (name.isEmpty) {
        None
    } else {
        try{
            Some((id.toInt, name.trim))
        } catch {
            case _: NumberFormatException => None
        }
    }
}.toDF("id","name")


//// 03 ////
val rawArtistAlias=spark.read.textFile("./Chap3_data/profiledata_06-May-2005/artist_alias.txt")
val artistAlias=rawArtistAlias.flatMap{line=>
    val Array(artist, alias)=line.split('\t')
    if (artist.isEmpty) {
        None
    } else{
        Some((artist.toInt, alias.toInt))
    }
}.collect().toMap

artistAlias.head

artistByID.filter($"id" isin (1208690,1003926)).show()





///////////////////////////////////
////// 3.4 첫 번째 모델 생성 ///////
///////////////////////////////////

//// 01 ////
import org.apache.spark.sql._
import org.apache.spark.broadcast._

def buildCounts(
    rawUserArtistData:Dataset[String],
    bArtistAlias: Broadcast[Map[Int,Int]]): DataFrame = {
        rawUserArtistData.map{line=>
        val Array(userID, artistID, count)=line.split(' ').map(_.toInt)
        val finalArtistID=
            bArtistAlias.value.getOrElse(artistID,artistID)
            (userID,finalArtistID,count)
        }.toDF("user","artist","count")
    }

val bArtistAlias = spark.sparkContext.broadcast(artistAlias)

val trainData=buildCounts(rawUserArtistData,bArtistAlias)
trainData.cache()


//// 02 ////
import org.apache.spark.ml.recommendation._
import scala.util.Random

val model=new ALS().
    setSeed(Random.nextLong()).
    setImplicitPrefs(true).
    setRank(10). //모델의 잠재요인 개수(사용자-특징 행렬, 제품-특징 행렬에서의 열 개수 k)
    setRegParam(0.01).//표준 과적합 파라미터.(Lambda)
    setAlpha(1.0).//행렬 분해 과정에서 관측된 사용자-제품, 관측되지 않은 사용자-제품 간의 상대적 가중치 조절
    setMaxIter(5).//행렬분해 반복 횟수
    setUserCol("user").
    setItemCol("artist").
    setRatingCol("count").
    setPredictionCol("prediction").
    fit(trainData)

model.userFactors.show(1,truncate=false)

////////////////////////////////
////// 3.5 추천 결과 추출 ///////
////////////////////////////////

//// 01 ////
val userID=2093760

val existingArtistIDs=trainData.
    filter($"user"===userID).
    select("artist").as[Int].collect()

artistByID.filter($"id" isin (existingArtistIDs:_*)).show()

//// 02 ////
def makeRecommendations(
    model: ALSModel,
    userID: Int,
    howMany: Int): DataFrame = {
    val toRecommend = model.itemFactors.
        select($"id".as("artist")).
        withColumn("user",lit(userID))

    model.transform(toRecommend).
        select("artist","prediction").
        orderBy($"prediction".desc).
        limit(howMany)
}

val topRecommendations = makeRecommendations(model, userID, 5)
spark.conf.set( "spark.sql.crossJoin.enabled" , "true" )
//이거 안해주면 아래 오류 뜸.

topRecommendations.show()

val recommendedArtistIDs = 
    topRecommendations.select("artist").as[Int].collect()

artistByID.filter($"id" isin (recommendedArtistIDs:_*)).show()

//////////////////////////////////////
////// 3.6 추천 품질 평가 : AUC ///////
//////////////////////////////////////

////01////

import scala.collection.mutable.ArrayBuffer
//평균 AUC 구현 출처 : Sandy Ryza's GITHUB
def areaUnderCurve(
      positiveData: DataFrame,
      bAllArtistIDs: Broadcast[Array[Int]],
      predictFunction: (DataFrame => DataFrame)): Double = {

    // 이 함수가 출력하는 것은 사용자별 AUC다. 그러므로 평균 AUC라고 볼 수 있다. 

    // 보류된 데이터를 "positive"로 간주한다.
    // score를 포함하여 각각에 대해 예측한다.
    val positivePredictions = predictFunction(positiveData.select("user", "artist")).
      withColumnRenamed("prediction", "positivePrediction")

    //BinaryClassificationMetrics.areaUnderROC는 실제로 많은 작은 AUC 문제가 있다.
    // 직접 계산이 가능할 때 비효율적이므로 여기서 사용되지 않는다.

    // 각 사용자에 대해 "negative" 데이터셋을 만든다. 이것은 사용자에게 "positive"였던 artist를 제외하고
    // 다른 아티스트 중에 랜덤으로 선택한 것이다.
    val negativeData = positiveData.select("user", "artist").as[(Int,Int)].
      groupByKey { case (user, _) => user }.
      flatMapGroups { case (userID, userIDAndPosArtistIDs) =>
        val random = new Random()
        val posItemIDSet = userIDAndPosArtistIDs.map { case (_, artist) => artist }.toSet
        val negative = new ArrayBuffer[Int]()
        val allArtistIDs = bAllArtistIDs.value
        var i = 0
        // 무한 루프를 피하기 위해 모든 아티스트에 대해 최대 한 번의 패스를 만든다.
        // 거기다 negative와 positive 사이즈가 같으면 종료
        while (i < allArtistIDs.length && negative.size < posItemIDSet.size) {
          val artistID = allArtistIDs(random.nextInt(allArtistIDs.length))
          // 새로운 ID만을 추가한다.
          if (!posItemIDSet.contains(artistID)) {
            negative += artistID
          }
          i += 1
        }
        // 사용자 ID가 다시 추가 된 세트 반환
        negative.map(artistID => (userID, artistID))
      }.toDF("user", "artist")

    // 나머지에 대한 예측:
    val negativePredictions = predictFunction(negativeData).
      withColumnRenamed("prediction", "negativePrediction")

    // 사용자별 negative 예측과 positive 예측만을 결합한다.
    // 그러면 각 사용자 내에서 가능한 모든 positive 및 negative 예측 쌍에 대한 행이 생성된다.
    val joinedPredictions = positivePredictions.join(negativePredictions, "user").
      select("user", "positivePrediction", "negativePrediction").cache()

    // Count the number of pairs 
    val allCounts = joinedPredictions.
      groupBy("user").agg(count(lit("1")).as("total")).
      select("user", "total")
    // Count the number of correctly ordered pairs per user
    val correctCounts = joinedPredictions.
      filter($"positivePrediction" > $"negativePrediction").
      groupBy("user").agg(count("user").as("correct")).
      select("user", "correct")

    // Combine these, compute their ratio, and average over all users
    val meanAUC = allCounts.join(correctCounts, Seq("user"), "left_outer").
      select($"user", (coalesce($"correct", lit(0)) / $"total").as("auc")).
      agg(mean("auc")).
      as[Double].first()

    joinedPredictions.unpersist()

    meanAUC
  }



///////////////////////////
////// 3.7 AUC 계산 ///////
//////////////////////////

//// 01 ////
val allData = buildCounts(rawUserArtistData, bArtistAlias)
val Array(trainData, cvData) = allData.randomSplit(Array(0.9,0.1))

trainData.cache()
cvData.cache()

val allArtistIDs = allData.select("artist").as[Int].distinct().collect()
val bAllArtistIDs = spark.sparkContext.broadcast(allArtistIDs)

val model = new ALS().
    setSeed(Random.nextLong()).
    setImplicitPrefs(true).
    setRank(10).setRegParam(0.01).setAlpha(1.0).setMaxIter(5).
    setUserCol("user").setItemCol("artist").
    setRatingCol("count").setPredictionCol("prediction").
    fit(trainData)

areaUnderCurve(cvData,bAllArtistIDs, model.transform)
/////////////////////////////
//AUC : 0.9019240126780086 //
/////////////////////////////

//// 03 ////
// 모든 사용자에게 가장 많이 재생된 아티스트 추천
// 개인화 되지 않음.
def predictMostListened(train:DataFrame)(allData: DataFrame)={
    val listenCounts = train.
        groupBy("artist").
        agg(sum("count").as("prediction")).
        select("artist","prediction")

    allData.
        join(listenCounts,Seq("artist"),"left_outer").
        select("user","artist","prediction")
}

areaUnderCurve(cvData, bAllArtistIDs, predictMostListened(trainData))
/////////////////////////////
//AUC : 0.8764492104739889 //
/////////////////////////////



//////////////////////////////////////
////// 3.8 하이퍼 파라미터 선택 ///////
//////////////////////////////////////
//// 01 ////
/*
Brute-force : 무작위 대입법
하이퍼 파라미터 조정하며, AUC 비교를 통한 최적 모델 선정
*/
val evaluations=
    for (rank <- Seq(5,30);
        regParam <-Seq(4.0, 0.0001);
        alpha <-Seq(1.0,40.0))
    yield {
        val model= new ALS().
        setSeed(Random.nextLong()).
        setImplicitPrefs(true).
        setRank(rank).setRegParam(regParam).
        setAlpha(alpha).setMaxIter(20).
        setUserCol("user").setItemCol("artist").
        setRatingCol("count").setPredictionCol("prediction").
        fit(trainData)

    val auc = areaUnderCurve(cvData,bAllArtistIDs,model.transform)

    model.userFactors.unpersist()
    model.itemFactors.unpersist()
    
    (auc, (rank, regParam, alpha))
    }

evaluations.sorted.reverse.foreach(println)
/*
alpha : 1일 때보다 항상 40일 때가 좋음.
    => 사용자 경향 : 듣지 않았던 아티스트보다 들었던 아티스트에 더 집중한다.

regParam : 과적합에 민감한 모델임 -> 높을수록 좋다.
rank : 모델의 잠재요인 개수 => 5개의 특징으로는 취향을 설명하기에 특징수가 매우 적음.
*/

////////////////////////////////
////// 3.9 추천 결과 생성 ///////
////////////////////////////////

//데이터에서 사용자 50명을 추출 후 이에 맞는 추천을 제공
val someUsers = allData.select("user").as[Int].dictinct().take(50)
val someRecommendations =
    someUsers.map(userID => (userID, makeRecommendations(model, userID, 5)))

someRecommendations.foreach{case (userID, recsDF)=>
    val recommendedArtists = recsDF.select("artist").as[Int].collect()
    println(s"$userID -> ${recommendedArtists.mkString(", ")}")
}
//mkString : 구분자 사용해서 하나의 string으로 생성

