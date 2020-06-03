#!/bin/bash

input_table_name="tmp.tmp_top_search_result"
output_s3_path="s3://vomkt-emr-rec/zn/data/search_data"
rep_num=20
spark-submit --name zn-data --deploy-mode client --master yarn --driver-cores 1 --driver-memory 2G --conf spark.debug.maxToStringFields=200 --conf spark.hadoop.mapred.output.compress=false --class com.vova.utils.Export2S3 s3://vomkt-emr-rec/dyshu/jar/spark-utils.jar ${input_table_name} ${output_s3_path} ${rep_num}
echo "end"


if [ $? -ne 0 ]; then
  exit 1
fi