import pymongo

'''
how to call: ml2imdb(1)
return 114709
'''
def ml2imdb(ml_id):
    ml_id = int(ml_id)
    client = pymongo.MongoClient("39.98.136.173", 9099)
    client.movie.authenticate('user','cloud',mechanism='SCRAM-SHA-1')
    database = client['movie']
    links = database['links_temp']
    imdbId = links.find_one({"movieId": ml_id})
    return imdbId['imdbId']