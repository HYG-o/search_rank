"""
搜索日志：tmp.tmp_top_search_result
用户信息：dws.dws_buyer_portrait
商品信息：ads.ads_goods_id_behave
select t1.key_words,t1.click,t1.first_cat_id,t1.second_cat_id,t1.brand_id,t1.absolute_position,t1.region_id,t1.gender,t1.platform,t1.birthday,t1.collector_tstamp,
t2.clicks,t2.impressions,t2.sales_order,t2.users,t2.impression_users,t2.payed_user_num,t2.gmv,t2.ctr,t2.gcr,t2.cr,t2.click_cr,t2.grr,t2.sor,t2.lgrr,t2.search_click,
t2.sales_order_m,t2.gender,t2.search_score,t2.score,t2.rate,t2.gr,t2.cart_uv,t2.cart_pv,t2.cart_rate,t2.shop_price,t2.show_price
from tmp.tmp_top_search_result as t1 join dws.dws_buyer_portrait as t2 on t1.buyer_id=t2.buyer_id and t2.pt='2020-06-02' limit 1;		# 日志 + 用户
from tmp.tmp_top_search_result as t1 join ads.ads_goods_id_behave as t2 on t1.goods_id=t2.goods_id limit 1;								# 日志 + 商品
hive -f join_hive_table.sh > zn_search_data.txt
aws s3 cp zn_search_data.txt s3://vomkt-emr-rec/zn/data/
"""
search_log_fields = ['collector_tstamp','birthday','buyer_id','key_words','click', 'absolute_position', 'gender', 'platform',  \
                     'first_cat_id','second_cat_id','brand_id','region_id']
good_fields = ['goods_id','brand_id','clicks','impressions','sales_order','users','impression_users','payed_user_num','gmv','ctr','gcr','cr','click_cr', \
'grr','sor','lgrr','search_click','sales_order_m','gender','search_score','score','rate','gr','cart_uv','cart_pv','cart_rate','shop_price','show_price',]
FIELDS = ['s-' + e for e in search_log_fields] + ['g-' + e for e in good_fields]
F2I = {e: i for i, e in enumerate(FIELDS)}
I2F = {i: e for i, e in enumerate(FIELDS)}

def get_hive_join_table_file(num=10):
    fields = ",".join(["t1." + str(e) for e in search_log_fields] + ["t2." + str(e) for e in good_fields])
    sql_cmd = "select " + fields + " from tmp.tmp_top_search_result as t1 join ads.ads_goods_id_behave as t2 on t1.goods_id=t2.goods_id limit " + str(num) + ";"
    print(sql_cmd)
    with open("hive_join_table.sh", "w", encoding="utf8") as fin:
        fin.write(sql_cmd)

if __name__ == "__main__":
    get_hive_join_table_file()