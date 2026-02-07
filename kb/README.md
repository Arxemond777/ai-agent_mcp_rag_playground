# This project uses WEB Flux + NETTY + reactor not spring boot + tomcat  

```bash
curl http://localhost:8080/hello
curl http://localhost:8080/stream
```


### Apache Jena Fuseki  (admin / admin)
```bash
# if ADMIN_PASSWORD not set -> docker logs fuseki and look for the randomly generated password
docker run -d \
  --name fuseki \
  -p 3030:3030 \
  -e ADMIN_PASSWORD=admin \
  stain/jena-fuseki

# create the dataset
curl -u admin:admin -X POST http://localhost:3030/$/datasets \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data "dbName=ds&dbType=tdb2"

```
UI → http://localhost:3030  
login: admin  
password: admin  

SPARQL endpoint → http://localhost:3030/ds  


### There is a simple demo of working with Neptune  (disclaimer for Neptune needs Aws4Signer, so I used Apache Jena Fuseki (Docker) for the local start-up)
```bash
curl -X POST http://localhost:8080/neptune/seed # populate some test data

curl http://localhost:8080/neptune/friends/Alice
curl http://localhost:8080/neptune/fof/Alice
curl -X DELETE http://localhost:8080/neptune/clear
```