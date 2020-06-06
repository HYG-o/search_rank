docker rm -f search_rank
docker kill search_rank
#docker build -t ic/searchrank:v$1 --no-cache .
docker build -t ic/searchrank:v$1 .
docker run -d --name search_rank -v /opt/searchranklog:/server/log -p 51658:51658 --net='host' ic/searchrank:v$1
#docker run -it --name search_rank -v /opt/searchranklog:/server/log -p 51658:51658 --net='host' ic/searchrank:v$1 bash


#docker run -it --name search_rank -v /opt/userhome/algo/searchranklog:/server/log -p 51658:51658 --net='host' ic/searchrank:v$1 bash
#docker exec -it search_rank
