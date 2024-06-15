import pymongo
from datetime import datetime
import os
from car_agent import AgentRunner

mongo_client = pymongo.MongoClient("mongodb://root:passwd@mongo:27017/")
db = mongo_client["car_posts"]
collection = db["posts"]
agentRunner = AgentRunner()

# Fetch the next post data
next_post = collection.find_one({
    "schedule_date": {"$lte": datetime.now().isoformat()},
    "approved": False
}, sort=[("schedule_date", pymongo.ASCENDING)])

if next_post:
    agent_response = agentRunner.run_agent(next_post['name'])

    # Save the generated post content to the filesystem
    post_filename = f"{next_post['name'].replace(' ', '_')}_post.txt"
    with open(post_filename, 'w') as file:
        file.write(agent_response)

    # Mark the post as approved
    collection.update_one({"_id": next_post["_id"]}, {"$set": {"approved": True}})

    print(f"Post for {next_post['name']} generated and saved as {post_filename}.")
else:
    print("No posts scheduled for today or all scheduled posts are approved.")