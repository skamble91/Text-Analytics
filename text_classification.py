#Code for Text Classification Using Pyspark 
# walmart analysis
from pyspark.sql import SQLContext, Row, HiveContext
from pyspark.sql.functions import col, udf, StringType, concat, lit
from pyspark.sql.types import *
from pyspark import SparkContext
import nltk
import re
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from string import digits
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics

sc = SparkContext()
hc = HiveContext(sc)

df=hc.sql("select *  from table where (col1 like '%text%') and col_2 <> 'text2") 
df_text=df.select('col1','col2','col3','col4','col5', 'col6', 'col7')

def output_y(col2,col3):
    if (col2=="null" or col2=='NULL' or len(col2)==0):
        if (col3==3 or col2==1):
            return 1.0
        else:
            return 0.0
    else:
        return 1.0


ol_val=udf(output_y,DoubleType())
data=df_text.withColumn("label", ol_val(df_text.col2,df_text.col2))

# indexerop1 = StringIndexer(inputCol="col4", outputCol="op1")
# indexerop2 = StringIndexer(inputCol="col5", outputCol="op2")

# indexedop1 = indexerop1.fit(data).transform(data)
# indexedop2 = indexerop2.fit(indexedop1).transform(indexedop1)

#text Cleaning
def cleaning_text(sentence):
   sentence=sentence.lower()
   sentence=re.sub('\'','',sentence)
   sentence=re.sub('^\d+\/\d+|\s\d+\/\d+|\d+\-\d+\-\d+|\d+\-\w+\-\d+\s\d+\:\d+|\d+\-\w+\-\d+|\d+\/\d+\/\d+\s\d+\:\d+',' ',sentence)# dates removed
   sentence=re.sub(r'(.)(\/)(.)',r'\1\3',sentence)
   sentence=re.sub("(.*?\//)|(.*?\\\\)|(.*?\\\)|(.*?\/)",' ',sentence)
   sentence = re.sub('[%s]' % re.escape(string.punctuation),'',sentence)
   stops = set(["a","about","all","an","become","beyond","do","etc","for","her","interest","mine","now","them","throughout","very","which","thanks",
   "above","almost","and","becomes","both","done","even","former","here","into","more","nowhere","perhaps","sixty","themselves","thru",
   "via","while","posrr","scorr","thankyou","across","alone","another","becoming","bottom","due","ever","formerly","hereafter",
   "is","moreover","of","please","so","then","thus","w","whither","posrt","scodi","along","any","been","but","during","every",
   "forty","hereby","it","most","often","put","some","thence","tl""was","who","posdi","scodu","after","already","anyhow","before",
   "by","each","everyone","fosm","herein","its","mostly","once","rather","somehow","there","tm","we","whoever","posdu","scord","afterwards",
   "also","anyone","beforehand"])
   cleaned=' '.join([w for w in sentence.split() if not w in stops])
   cleaned=' '.join([w for w in cleaned.split() if not len(w)<2 ])
   cleaned=cleaned
   if(len(cleaned)<=1):
      return "NA"
   else:
      return cleaned



org_val=udf(cleaning_text,StringType())
data=data.withColumn("cleaned",org_val(data.summary))
data = data.filter(data["cleaned"]!= "NA")

tokenizer = Tokenizer(inputCol="cleaned", outputCol="words")
wordsData = tokenizer.transform(data)
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
featurizedData = hashingTF.transform(wordsData)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# assembler=VectorAssembler(inputCols=['op1','op2','op3','pr1','features'],outputCol='FeaturesFinal')
# assembled=assembler.transform(rescaledData)

ADS=rescaledData.select('label','features')

# (trainingData, testData) = ADS.randomSplit([0.7, 0.3],100)

lp=ADS.select(col("label"), col("features")).map(lambda row: LabeledPoint(row.label, row.features))
lp=ADS.select(col("wm_record_id"),col("label"), col("features")).map(lambda row: LabeledPoint(row.label, row.features))

modellr = LogisticRegressionWithLBFGS.train(lp, iterations=100)
modellr.clearThreshold()
modellr.setThreshold(0.7)

model_path= os.path.join('/user/name', 'models', 'log_reg_text')
modellr.save(sc,model_path)

predictionAndLabelspoint65 = testData.map(lambda p: (float(modellr.predict(p.FeaturesFinal)), p.label))
# modellr.clearThreshold()
# modellr.setThreshold(0.67)

total=test_data.count()
correct = predictionAndLabelspoint65.filter(lambda (predicted, actual): predicted == actual).count()
count_1_1 = predictionAndLabelspoint65.filter(lambda (predicted, actual): predicted == actual==1.0).count()
count_0_1 = predictionAndLabelspoint65.filter(lambda (predicted, actual): predicted == 1.0 and actual==0.0).count()
count_1_0 = predictionAndLabelspoint65.filter(lambda (predicted, actual): predicted == 0.0 and actual==1.0).count()
count_0_0 = predictionAndLabelspoint65.filter(lambda (predicted, actual): predicted == actual==0.0).count()

accuracy = float(correct)/ float(total)
misclassificationrate=float(count_0_1+count_1_0)/float(total)
truenegatives=float(count_0_0)/float((count_1_0+count_0_0))
falsepositiverate=float(count_0_1)/float(count_0_0+count_0_1)
specificity=float(count_0_0)/float(count_0_0+count_1_0)
precision=float(count_1_1)/float(count_0_1+count_1_1)
falsenegrate=float(count_1_0)/float((count_1_0+count_0_0))

print("\n\n Accuray = %s \n\n"%(accuracy*100))
print("\n\n misclassificationrate = %s \n\n"%(misclassificationrate*100))
print("\n\n truepositiverate = %s \n\n"%(truepositiverate*100))
print("\n\n falsepositiverate = %s \n\n"%(falsepositiverate*100))
print("\n\n specificity = %s \n\n"%(specificity*100))
print("\n\n precision = %s \n\n"%(precision*100))
print("\n\n falsenegative = %s \n\n"%(falsenegrate*100))

print ("\n\n 1 1 : %s"%(count_1_1))
print (" 1 0 : %s"%(count_1_0))
print (" 0 1 : %s"%(count_0_1))
print (" 0 0 : %s \n\n"%(count_0_0)) 

metrics = BinaryClassificationMetrics(predictionAndLabelspoint65)
print("Area under PR = %s" % metrics.areaUnderPR)
print("Area under ROC = %s" % metrics.areaUnderROC)
