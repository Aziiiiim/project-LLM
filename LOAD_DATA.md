Loading data into the Neo4j database can be a bit tricky. I'll then provide you the steps to do it.

References: 
- https://neo4j.com/docs/operations-manual/current/docker/dump-load/
- https://neo4j.com/docs/getting-started/appendix/example-data/

First of all, we need to create the data volume (in which we'll load the data)

```
docker compose create
```

Once the volume and the container are created, we can load the data using the following command.

💡INFO: the database needs to be shut down to load the data, that's why we can't just run the command within the container  
💡INFO: In the Community Edition, we only have one database, named "neo4j". The dump file must then have the same name.

Make sure the paths and names in the command match yours.

```
docker run -it --rm \
-v project-llm_neo4j_data:/data \
-v ./neo4j_init:/neo4j_init \
neo4j \
neo4j-admin database load neo4j --from-path=/neo4j_init --overwrite-destination=true --verbose
```

Now, we can get the container back up and running with ```docker compose up -d```

Finally, start the container and make sure the data is loaded correctly. 
- Connect to the database with the credentials set in the docker-compose / env file
- Check the data by running this command in the web console:

```
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100
```