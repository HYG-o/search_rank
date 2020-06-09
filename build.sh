docker rm -f search_rank
 docker kill search_rank
 #docker build -t searchrank:v$1 --no-cache .
 docker build -t searchrank:v$1 .
 #docker run -d --name search_rank -v /opt/searchranklog:/server/log -p 51658:51658 searchrank:v$1
 docker run -it --name search_rank -v /opt/searchranklog:/server/log -p 51658:51658 --net='host' searchrank:v$1 bash

 #docker run -it --name search_rank -v /opt/searchranklog:/server/log -p 51658:51658 --net='host' searchrank:v$1 bash
 #docker exec -it search_rank