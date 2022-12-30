#Veriler Flo mağazacılığa aittir. Verilerin  paylaşımına izin verilememektedir.
#Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
#olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.
#12  Değişken 19.945 gözlem

#master_id Eşsiz müşteri numarası
#order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
#last_order_channel En son alışverişin yapıldığı kanal
#first_order_date Müşterinin yaptığı ilk alışveriş tarihi
#last_order_date Müşterinin yaptığı son alışveriş tarihi
#last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
#last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
#order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
#order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
#customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
#customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
#interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listes

#Kütüphaneler

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import datetime
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_ = pd.read_csv("C:/Users/sfkul/Downloads/FLO_RFM_Analizi/FLO_RFM_Analizi/flo_data_20k.csv")
df = df_.copy()
#ilk 10 gözlem
df.head(10)
#Değişkenlerin ismi
df.columns

#boyut
df.shape

#Betimsel istatistik
df.describe().T

#Boş değerler
df.isnull().sum()

#. Değişken tipleri, incelemesi
df.info()

#Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
#alışveriş sayısı ve harcaması için yeni değişkenler
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] =df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df.head()

#Değişken tiplerini inceleyelim. Tarih ifade eden değişkenlerin tipini date'e çevirelim

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] =df[date_columns].apply(pd.to_datetime)
df.info()

#Alışveriş kanallarındaki müşteri sayısının, toplam ürün sayısı ve toplam harcamalarının dağılımı

df.groupby("order_channel").agg({"master_id": "count",
                                "order_num_total": "sum",
                                 "customer_value_total": "sum"})

#En  fazla  kazancı getiren ilk 10 müşteri

df.sort_values("customer_value_total", ascending=False)[:10]

#En fazla sipariş veren ilk 10 müşteri

df.sort_values("order_num_total", ascending=False)[:10]

#Veri ön hazırlık sürelerini fonksiyonlaştıralım

def data_prep(dataframe):
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[data_columns] = dataframe[date_columns].apply(pd.to_datetime)
    return df

#RFM Metriklerin Hesaplanması

#Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi

df["last_order_date"].max() #2021-05-30
analysis_date = dt.datetime(2021,6,1)

#customer_id, recency, frequnecy ve monetary değerlerinin yer aldığı yeni bir rfm dataframe oluşturalım

rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

rfm.head()

#RF ve RFM Skorlarının Hesaplanması (Calculating RF and RFM Scores)

#Recency, Frequency ve Monetary öetriklerini qcut yardımı ile 1-5 arasında skorlara çevrilmesi
# bu skorları recency_score,fruquency_score ve monetary_score olarak kaydedilmesi

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1] )
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=(1, 2, 3, 4, 5))

rfm.head()

#recency_score ve frequency_score'u tek bir değişken olarak ifade edilmesi ve RF_SCORE olarak kaydedilmesi

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
rfm.head()

#RF Skorlarını Segment Olarak Tanımlaması

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm.head()

#Segmentlerin recency, frequency ve monetary ortamlarını inceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak iletişime geçeilmek isteniliyor. Bu müşterilerin sadık  ve
# kadın kategorisinden alışveriş yapan kişiler olması planlandı. Müşterilerin id numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs
# olarak kaydediniz.

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

rfm.head()

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte iyi müşterilerden olan ama uzun süredir
# alışveriş yapmayan ve yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating","new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & ((df["interested_in_categories_12"].str.contains("ERKEK"))|(df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)

