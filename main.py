import os 
import sys 
"""
İşletim sistemiyle etkileşim kurmak için kullanılır. 
Ortam değişkenlerini ayarlamak için

Python çalışma ortamı ve yollarını yönetmek için kullanılır. 
Özellikle Spark'ın Python yollarını Python’a tanıtmak için gereklidir.
"""

# Java ve Spark yolları
os.environ['JAVA_HOME'] = 'C:\\Program Files\\Java\\jdk-11.0.17'
os.environ['SPARK_HOME'] = 'C:\\spark\\spark-3.5.5-bin-hadoop3'
os.environ['PYSPARK_PYTHON'] = 'C:\\Python310\\python.exe'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:\\Python310\\python.exe'
# Ortam değişkenleri

# Python yolları
sys.path.append('C:\\spark\\spark-3.5.5-bin-hadoop3\\python')
sys.path.append('C:\\spark\\spark-3.5.5-bin-hadoop3\\python\\lib\\py4j-0.10.9.5-src.zip')
# Spark ile Python arasında iletişim kuran bir köprü görevi gören Py4J kütüphanesinin sıkıştırılmış dosyası. 
# PySpark, Java’da çalışan Spark çekirdeğiyle Python arasında veri alışverişi için Py4J’yi kullanır.
# Spark ve Py4J kütüphanesi gibi Spark-Python köprüsü bileşenlerinin Python tarafından bulunabilmesini sağlamak için kullanılr.

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from datetime import datetime
"""
Spark ile Python arasında bağlantı kurmak ve veri işlemlerini yönetmek için kullanılan temel yapı taşını temsil eder. 
SparkSession, Spark’ın bütün bileşenlerine (SQL, MLlib, GraphX vb.) erişimi sağlar.

StringIndexer: Kategorik (metinsel) verileri sayısal verilere dönüştürmek için kullanılır. 
Makine öğrenmesi modelleri sayısal verilerle çalıştığı için bu dönüşüm çok önemlidir.
VectorAssembler: Birden fazla özelliği (feature) birleştirerek tek bir vektör haline getirir. 
Makine öğrenmesi algoritmaları bu vektörleri kullanarak tahminleme yapar.

RandomForestClassifier: Verilen özelliklere göre hangi sınıfa (örneğin bitki yetişme durumu) ait olduğunu tahmin eder.

Modelin başarısını ölçmek için kullanılan araç. 
MulticlassClassificationEvaluator, birden fazla sınıfa sahip etiketlerin tahmin performansını ölçer.

Sınıflandırma sonuçlarının detaylı değerlendirmesini sağlar. 
Modelin sınıflandırma performansını analiz eder.

Tarih ve zaman bilgisini almak için kullanılır.
İşlem süresini ölçmek, sonuçların kaydedilmesi, zaman damgası (timestamp) eklemek için kullanılır.
"""

# Spark oturumu başlatılır.
spark = SparkSession.builder \
    .appName("Bitki Yetişme Analizi") \
    .master("local[*]") \
    .getOrCreate()
"""
Spark’ın çekirdeğini başlatır ve PySpark ortamını aktif hale getirir.
builder: Spark oturumunu yapılandırmak için kullanılır.
Spark’ın lokal modda çalışacağını belirtir.
Eğer önceden bir Spark oturumu varsa onu getirir, yoksa yeni bir tane oluşturur.
"""

try:
    # Veri seti okunur.
    data = spark.read.csv("veri_seti.csv", header=True, inferSchema=True)
    """
    Spark DataFrame olarak CSV dosyasını okur.
    Spark’ın sütun tiplerini otomatik olarak algılamasını sağlar.
    """
    # Eksik değerler temizlenir.
    data = data.dropna()

    # Ürün bazlı ortalama analiz 
    print("\n[GRUP ANALİZİ] Ürün Bazlı Ortalama Değerler:")
    data.groupBy("label").avg("N", "P", "K", "temperature", "humidity", "ph", "rainfall").show(truncate=False)
    """
    “label” sütununa göre gruplama yapar ve özelliklerin ortalamasını alır.
    Ürünlern genel karaktristik özelliklerini analiz ederiz.
    """

    print("\nVeri Seti Önizleme:")
    data.show(5)
    """
    Verinin genel yapısını ön izleme amacıyla Veri setinin ilk 5 satırını gösterir.
    """

    # Veri tiplerini gösterir.
    print("\nVeri Seti Şeması:")
    data.printSchema()

    # Label'ı (ürün ismi) sayısal değerlere çevir
    indexer = StringIndexer(inputCol="label", outputCol="label_index")
    indexer_model = indexer.fit(data)
    data_indexed = indexer_model.transform(data)
    """
    “label” sütununu sayısal değerlere dönüştürür.
    Modeli oluşturur.
    Veriyi dönüştürür.
    """

    # Özellik sütunları 
    feature_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(data_indexed)
    """
    Özellik sütunları tek bir vektör haline getirilir.
    “features” sütunu, modelin girdi vektörünü temsil eder.
    """

    # Eğitim (%80) ve test (%20) 
    train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=42)
    # Veri seti rastgele ayrılır, sonuçlar tekrarlanabilir.

    # Random Forest modeli
    rf = RandomForestClassifier(featuresCol="features", labelCol="label_index", numTrees=100)
    model = rf.fit(train_data)
    """
    Modelin giriş vektörünü (features) alır.
    Modelin tahmin etmeye çalıştığı sınıf sütunu (label_index).
    100 adet karar ağacı
    Modeli eğitim verisiyle eğitir.
    """
    # Test verisi üzerinde tahminler yapılır.
    predictions = model.transform(test_data)

    # Modelin başarısı değerlendirilirç
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    # Etiket, tahmin edilen etiket, doğruluk oranı

    predictionAndLabels = predictions.select("prediction", "label_index") \
                                    .rdd.map(lambda x: (float(x[0]), float(x[1])))
    """
    Tahmin ve gerçek etiket çiftlerini bir RDD’ye dönüştürür.
    MulticlassMetrics kütüphanesi RDD formatında veri ister.
    """

    # Modelin sınıflandırma performansını analiz eder.
    metrics = MulticlassMetrics(predictionAndLabels)

    print("\nModel Performans Metrikleri:")
    print(f"Model Doğruluk Oranı: {accuracy:.2f}")
    print(f"Precision: {metrics.weightedPrecision:.2f}")
    print(f"Recall: {metrics.weightedRecall:.2f}")
    print(f"F1 Score: {metrics.weightedFMeasure():.2f}")

    # Kullanıcıdan veri alıp modelle tahmin yapılır.
    print("\nYeni Bitki Tahmini İçin Veri Girişi:")
    try:
        while True:
            try:
                N = input("N (Azot değeri, 0-140): ").strip()
                if not N:  # Boş giriş kontrolü
                    print("Lütfen bir değer girin!")
                    continue
                N = int(N)
                if not (0 <= N <= 140):
                    print("N değeri 0-140 arasında olmalıdır!")
                    continue
                break
            except ValueError:
                print("Lütfen geçerli bir sayı girin!")

        while True:
            try:
                P = input("P (Fosfor değeri, 5-145): ").strip()
                if not P:
                    print("Lütfen bir değer girin!")
                    continue
                P = int(P)
                if not (5 <= P <= 145):
                    print("P değeri 5-145 arasında olmalıdır!")
                    continue
                break
            except ValueError:
                print("Lütfen geçerli bir sayı girin!")

        while True:
            try:
                K = input("K (Potasyum değeri, 5-205): ").strip()
                if not K:
                    print("Lütfen bir değer girin!")
                    continue
                K = int(K)
                if not (5 <= K <= 205):
                    print("K değeri 5-205 arasında olmalıdır!")
                    continue
                break
            except ValueError:
                print("Lütfen geçerli bir sayı girin!")

        while True:
            try:
                temperature = input("Sıcaklık (°C, 8-44): ").strip()
                if not temperature:
                    print("Lütfen bir değer girin!")
                    continue
                temperature = float(temperature)
                if not (8 <= temperature <= 44):
                    print("Sıcaklık 8-44°C arasında olmalıdır!")
                    continue
                break
            except ValueError:
                print("Lütfen geçerli bir sayı girin!")

        while True:
            try:
                humidity = input("Nem (%, 14-100): ").strip()
                if not humidity:
                    print("Lütfen bir değer girin!")
                    continue
                humidity = float(humidity)
                if not (14 <= humidity <= 100):
                    print("Nem 14-100% arasında olmalıdır!")
                    continue
                break
            except ValueError:
                print("Lütfen geçerli bir sayı girin!")

        while True:
            try:
                ph = input("pH (0-14): ").strip()
                if not ph:
                    print("Lütfen bir değer girin!")
                    continue
                ph = float(ph)
                if not (0 <= ph <= 14):
                    print("pH 0-14 arasında olmalıdır!")
                    continue
                break
            except ValueError:
                print("Lütfen geçerli bir sayı girin!")

        while True:
            try:
                rainfall = input("Yağış (mm, 20-300): ").strip()
                if not rainfall:
                    print("Lütfen bir değer girin!")
                    continue
                rainfall = float(rainfall)
                if not (20 <= rainfall <= 300):
                    print("Yağış 20-300mm arasında olmalıdır!")
                    continue
                break
            except ValueError:
                print("Lütfen geçerli bir sayı girin!")

        user_input = spark.createDataFrame([(N, P, K, temperature, humidity, ph, rainfall)], 
            ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        """
        Spark DataFrame, verilerin büyük ölçekli analizini mümkün kılar. 
        Spark MLlib modelleriyle uyumlu veri yapısı sağlar.
        """

        # Ham veriyi modelin anlayacağı formata dönüştürme aşaması
        # Kullanıcının girdiği tüm özellikleri tek bir vektör haline getirir.
        user_input_assembled = assembler.transform(user_input)

        prediction = model.transform(user_input_assembled)
        predicted_label = prediction.select("prediction").collect()[0]["prediction"]
        """
        Model, veriyi işler ve tahmini prediction DataFrame'inde saklar.
        Modelin tahmin ettiği etiket numarası
        Tahmini etiket sütunu alınır.
        İlk satırdaki tahmin sonucu çekilir.
        Tahmin edilen bitki türü numarası elde edilir.
        """

        # Etiketleri bitki adına çevirme 
        labels = indexer_model.labels
        predicted_crop = labels[int(predicted_label)]

        print(f"\nTahmin Edilen Bitki Türü: {predicted_crop.upper()}")

        # Sonuçları, tarih ve saat bilgileriyle kaydeder.
        os.makedirs("sonuc", exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"sonuc/tahmin_{now}.txt"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Tahmin Edilen Bitki Türü: {predicted_crop.upper()}\n")
            f.write(f"Model Doğruluk Oranı: {accuracy:.2f}\n")
            f.write(f"Precision: {metrics.weightedPrecision:.2f}\n")
            f.write(f"Recall: {metrics.weightedRecall:.2f}\n")
            f.write(f"F1 Score: {metrics.weightedFMeasure():.2f}\n")
            f.write("\nGirilen Değerler:\n")
            f.write(f"N: {N}\nP: {P}\nK: {K}\n")
            f.write(f"Sıcaklık: {temperature}°C\n")
            f.write(f"Nem: {humidity}%\n")
            f.write(f"pH: {ph}\n")
            f.write(f"Yağış: {rainfall}mm\n")

            """
            metrics.weightedPrecision
            Modelin Precision (Kesinlik) metriğinin ağırlıklı ortalamasını verir.
            Doğru pozitif tahminlerin toplam pozitif tahminlere oranıdır. Yani, modelin “pozitif” dediği örneklerin ne kadarının gerçekten doğru olduğunu gösterir.

            metrics.weightedRecall
            Modelin Recall (Duyarlılık) metriğinin ağırlıklı ortalamasıdır.
            Doğru pozitiflerin toplam gerçek pozitiflere oranıdır. 
            Modelin gerçek pozitifleri ne kadar iyi yakaladığını gösterir.

            metrics.weightedFMeasure()
            Modelin F1 Score yani Precision ve Recall'un harmonik ortalamasıdır.
            F1 Score, modelin genel performansını dengeli şekilde gösterir.
            """

        print(f"\nTahmin sonucu '{output_path}' dosyasına yazıldı.")

    except ValueError as ve:
        print(f"\nHata: {str(ve)}")
    except Exception as e:
        print(f"\nBeklenmeyen bir hata oluştu: {str(e)}")

except Exception as e:
    print(f"\nProgram hatası: {str(e)}")
finally:
    # Spark oturumu kapatılır.
    spark.stop()